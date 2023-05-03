import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import os
from PIL import Image

from gmflow.gmflow import GMFlow
from utils.utils import (build_dataflow, get_augmentor, accuracy)
from utils.video_dataset import VideoDataSet
from utils.dataset_config import get_dataset_config
from utils.dataset_config import DATASET_CONFIG
from torch.utils.data.distributed import DistributedSampler
from utils.flow_viz import save_vis_flow_tofile, flow_to_image
from models.utilityNet import I3Du, resnet101
from models.budgetNet import I3Db
import torch.distributed as dist

from tqdm import tqdm
import csv

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')


        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation dataset')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='loss weight')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')

    # inference on a directory
    parser.add_argument('--datadir', metavar='DIR', help='path to dataset file list')
    parser.add_argument('--dataset', default='st2stv2',
                        choices=list(DATASET_CONFIG.keys()), help='which dataset.')
    parser.add_argument('--scale_range', default=[240, 250], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='spatial size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, '
                             'directly crop the input_size from center.')
    parser.add_argument('--threed_data', action='store_true',
                        help='format data to 5D for 3D onv.')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
                        choices=['rgb', 'flow'])
    parser.add_argument('--random_sampling', action='store_true',
                        help='[Uniform sampling only] perform non-deterministic frame sampling '
                             'for data loader during the evaluation.')
    parser.add_argument('--dense_sampling', action='store_true',
                        help='perform dense sampling for data loader')
    parser.add_argument('--augmentor_ver', default='v1', type=str, choices=['v1', 'v2'],
                        help='[v1] TSN data argmentation, [v2] resize the shorter side to `scale_range`')
    parser.add_argument('--groups', default=16, type=int, help='number of frames')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='[uniform sampling] number of frames per group; '
                             '[dense sampling]: sampling frequency')
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10],
                        help='[Test.py only] number of crops.')
    parser.add_argument('--num_clips', default=1, type=int,
                        help='[Test.py only] number of clips.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 18)')
    parser.add_argument('--lr_scheduler', default='multisteps', type=str,
                        help='learning rate scheduler',
                        choices=['step', 'multisteps', 'cosine', 'plateau'])
    parser.add_argument('--lr_steps', default=[50, 100, 150], type=float, nargs="+",
                        metavar='LRSteps', help='[step]: use a single value: the periodto decay '
                                                'learning rate by 10. '
                                                '[multisteps] epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='enable nesterov momentum optimizer')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')

    




    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

    # predict on sintel and kitti test set for submission
    parser.add_argument('--submission', action='store_true',
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')

    return parser


def main():

    args.distributed = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.distributed = True
    # device = torch.device("cuda", args.local_rank)


    # model
    encode_model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)
    action_recog_model = I3Du(num_classes=8).to(device)
    # action_recog_model = resnet101().to(device)
    privacy_model = I3Du().to(device)
    privacy_image_model = I3Db().to(device)

    
    # torch.distributed.init_process_group(backend="nccl")

    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)  # rank % torch.cuda.device_count()
        torch.backends.cudnn.enabled=False
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        encode_model = torch.nn.parallel.DistributedDataParallel(
            encode_model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        action_recog_model = torch.nn.parallel.DistributedDataParallel(
            action_recog_model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        privacy_model = torch.nn.parallel.DistributedDataParallel(
            privacy_model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)

    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            torch.backends.cudnn.enabled=False
            encode_model = torch.nn.DataParallel(encode_model)
            action_recog_model = torch.nn.DataParallel(action_recog_model)
            privacy_model = torch.nn.DataParallel(privacy_model)

            model_without_ddp = encode_model.module
        else:
            model_without_ddp = encode_model



    num_params = sum(p.numel() for p in encode_model.parameters())
    print('Number of params:', num_params)
    if not args.eval and not args.submission and args.inference_dir is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(encode_model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        encode_model.load_state_dict(weights, strict=args.strict_resume)
        # encode_model = torch.nn.parallel.DistributedDataParallel(
        #     encode_model.to(device),
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank)
        # encode_model = encode_model.to(device)
        # encode_model = torch.nn.DataParallel(encode_model)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        # print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))
    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset)
    
    # Data loading code
    val_list = os.path.join(args.datadir, val_list_name)

    norm_value= 255

    val_augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=[110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value],
                                  std=[38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value], disable_scaleup=args.disable_scaleup,
                                  threed_data=args.threed_data,
                                  is_flow=True if args.modality == 'flow' else False,
                                  version=args.augmentor_ver)

    val_dataset = VideoDataSet(args.datadir, val_list, args.groups, args.frames_per_group,
                               num_clips=args.num_clips,
                               modality=args.modality, image_tmpl=image_tmpl,
                               dense_sampling=args.dense_sampling,
                               transform=val_augmentor, is_train=False, test_mode=False,
                               seperator=filename_seperator, filter_video=filter_video)
    
    # valid_sampler = DistributedSampler(val_dataset)

    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                workers=args.workers,
                                is_distributed=args.distributed, epoch=0)


    train_list = os.path.join(args.datadir, train_list_name)

    train_augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=[110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value],
                                  std=[38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value],
                                    disable_scaleup=args.disable_scaleup,
                                    threed_data=args.threed_data,
                                    is_flow=True if args.modality == 'flow' else False,
                                    version=args.augmentor_ver)

    # train_sampler = DistributedSampler(train_dataset)
    train_dataset = VideoDataSet(args.datadir, train_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=train_augmentor, is_train=True, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                  workers=args.workers, is_distributed=args.distributed, epoch=0)
    

    train_criterion = nn.CrossEntropyLoss().to(device)

    ctr = 0
    save_dest = 'test4'
    total_epochs = 50

    '''
    1. Train the Target Network
    '''

    params_t = list(action_recog_model.parameters())
    optimizer_t = torch.optim.SGD(params_t, 0.01, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler_t = lr_scheduler.CosineAnnealingLR(optimizer_t, T_max= 60, eta_min=1e-7, verbose=True)

    for epoch in range(0, 60):
        print('\n Pre Training Action Recognition Now '+ str(epoch))

        '''
        Train action model (target) Type = 'T'
        '''

        trainT_top1, trainT_top5, train_losses = train(encode_model, action_recog_model, train_loader, optimizer_t, train_criterion, 'T', device)

        if args.distributed:
            dist.barrier()

        print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}'.format(epoch + 1, total_epochs, train_losses, trainT_top1), flush=True)
            
        valT_top1, valT_top5, val_losses = validate(encode_model, action_recog_model, val_loader, optimizer_t, train_criterion, 'T', device)

        if args.distributed:
            dist.barrier()

        print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\t'.format(epoch + 1, total_epochs, val_losses, valT_top1),flush=True)
        
        scheduler_t.step()

    '''
    2. Train the Encoder Network
    '''

    params_en = list(encode_model.parameters())
    optimizer_en = torch.optim.AdamW(params_en, lr=args.lr, weight_decay=args.weight_decay)
    scheduler_en = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_en, 0.01,
        args.num_steps + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=-1,
    )

    train_logger_pre_e = Logger(save_dest+'/'+'enc_train_pre'+'.log',['epoch','prec1_T','prec1_P','loss'])
    val_logger_pre_e = Logger(save_dest+'/'+'enc_val_pre'+'.log',['epoch','prec1_T','prec1_P','loss'])

    for epoch in range(0, 100):
        print('\n Pre Training Encoder Now '+ str(epoch))

        # more efficient zero_grad
        trainT_top1, trainT_top5, trainP_top1, trainP_top5, train_losses = train_en(encode_model, action_recog_model, privacy_model, train_loader, optimizer_en_t, train_criterion, device)
        train_logger_pre_e.log({'epoch': epoch, 'prec1_T': trainT_top1.item(), 'prec1_P': trainP_top1.item(), 'loss': train_losses})

        if args.distributed:
            dist.barrier()

        print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopP@1: {:.4f}\t'.format(
                    epoch + 1, total_epochs, train_losses, trainT_top1, trainP_top1), flush=True)
            

        valT_top1, valT_top5, valP_top1, valP_top5, val_losses = validate_en(encode_model, action_recog_model, privacy_model, val_loader, optimizer_en, train_criterion, device)
        val_logger_pre_e.log({'epoch': epoch, 'prec1_T': valT_top1.item(), 'prec1_P': valP_top1.item(), 'loss': val_losses})

        if args.distributed:
            dist.barrier()
        
        print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopP@1: {:.4f}\t'.format(
                    epoch + 1, total_epochs, val_losses, valT_top1, valP_top1),flush=True)
        scheduler_en.step()
            
        


    for ctr in range(10):
        train_logger_e = Logger(save_dest+'/'+'enc_train_'+str(ctr)+'.log',['epoch','prec1_T','prec1_P','loss'])
        val_logger_e = Logger(save_dest+'/'+'enc_val_'+str(ctr)+'.log',['epoch','prec1_T','prec1_P','loss'])

        train_logger_t = Logger(save_dest+'/'+'tar_train_'+str(ctr)+'.log',['epoch','prec1_T','loss'])
        val_logger_t = Logger(save_dest+'/'+'tar_val_'+str(ctr)+'.log',['epoch','prec1_T','loss'])

        train_logger_p = Logger(save_dest+'/'+'pri_train_'+str(ctr)+'.log',['epoch','prec1_T','loss'])
        val_logger_p = Logger(save_dest+'/'+'pri_val_'+str(ctr)+'.log',['epoch','prec1_T','loss'])

        train_logger_pi = Logger(save_dest+'/'+'prii_train_'+str(ctr)+'.log',['epoch','prec1_T','loss'])
        val_logger_pi = Logger(save_dest+'/'+'prii_val_'+str(ctr)+'.log',['epoch','prec1_T','loss'])


        params_en = list(encode_model.parameters())
        optimizer_en = torch.optim.AdamW(params_en, lr=args.lr, weight_decay=args.weight_decay)
        scheduler_en = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_en, args.lr,
            args.num_steps + 10,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=-1,
        )
        # optimizer_en = torch.optim.SGD(params_en, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        # scheduler_en = lr_scheduler.CosineAnnealingLR(optimizer_en, T_max= total_epochs, eta_min=1e-7, verbose=True)


        params_t = list(action_recog_model.parameters())
        optimizer_t = torch.optim.SGD(params_t, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        scheduler_t = lr_scheduler.CosineAnnealingLR(optimizer_t, T_max= total_epochs, eta_min=1e-7, verbose=True)


        params_p = list(privacy_model.parameters())
        optimizer_p = torch.optim.SGD(params_p, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        scheduler_p = lr_scheduler.CosineAnnealingLR(optimizer_p, T_max= total_epochs, eta_min=1e-7, verbose=True)
        
        params_en_t = list(encode_model.parameters()) + list(action_recog_model.parameters())
        optimizer_en_t = torch.optim.SGD(params_en_t, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        scheduler_en_t = lr_scheduler.CosineAnnealingLR(optimizer_en_t, T_max= total_epochs, eta_min=1e-7, verbose=True)


        '''
        Train Target
        '''
        
        for epoch in range(0, total_epochs):
            print('\n Fixing Encoder Training Action Recognition Now '+ str(epoch))

            '''
            Train action model (target)
            '''

            trainT_top1, trainT_top5, train_losses = train(encode_model, action_recog_model, train_loader, optimizer_t, train_criterion, 'T', device)

            train_logger_t.log({'epoch': epoch,'prec1_T': trainT_top1.item(), 'loss': train_losses})
            
            
            # train_logger_t.log({'epoch': epoch,'prec1_T': trainT_top1, 'loss': train_losses})

            if args.distributed:
                dist.barrier()

                
            valT_top1, valT_top5, val_losses = validate(encode_model, action_recog_model, val_loader, optimizer_t, train_criterion, 'T', device)
            
            
            # val_logger_t.log({'epoch': epoch,'prec1_T': valT_top1, 'loss': val_losses})
            val_logger_t.log({'epoch': epoch,'prec1_T': valT_top1.item(), 'loss': val_losses})

            if args.distributed:
                dist.barrier()

            scheduler_t.step()

            print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}'.format(
                        epoch + 1, total_epochs, train_losses, trainT_top1), flush=True)
            # print(torch.distributed.get_rank())
                
            print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\t'.format(epoch + 1, total_epochs, val_losses, valT_top1),flush=True)

        
        '''
        Train Privacy
        '''
        
        for epoch in range(0, total_epochs):
            print('\n Fixing Encoder Training Actor Recognition Now '+ str(epoch))

            '''
            Train actor model (privacy)
            '''

            trainT_top1, trainT_top5, train_losses = train(encode_model, privacy_model, train_loader, optimizer_p, train_criterion, 'P', device)

            train_logger_p.log({'epoch': epoch,'prec1_T': trainT_top1.item(), 'loss': train_losses})
            if args.distributed:
                dist.barrier()

                
            valT_top1, valT_top5, val_losses = validate(encode_model, privacy_model, val_loader, optimizer_p, train_criterion, 'P', device)


            val_logger_p.log({'epoch': epoch,'prec1_T': valT_top1.item(), 'loss': val_losses})
            if args.distributed:
                dist.barrier()

            scheduler_p.step()

            print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}'.format(
                        epoch + 1, total_epochs, train_losses, trainT_top1), flush=True)
                
            print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\t'.format(epoch + 1, total_epochs, val_losses, valT_top1),flush=True)
        
        
        
        '''
        Train Encoder
        '''
        for epoch in range(0, 50):
            print('\n Training Encoder Now '+ str(epoch))

            # more efficient zero_grad
            trainT_top1, trainT_top5, trainP_top1, trainP_top5, train_losses = train_en(encode_model, action_recog_model, privacy_model, train_loader, optimizer_en_t, train_criterion, device)
            train_logger_e.log({'epoch': epoch, 'prec1_T': trainT_top1.item(), 'prec1_P': trainP_top1.item(), 'loss': train_losses})
            if args.distributed:
                dist.barrier()

            valT_top1, valT_top5, valP_top1, valP_top5, val_losses = validate_en(encode_model, action_recog_model, privacy_model, val_loader, optimizer_en, train_criterion, device)
            val_logger_e.log({'epoch': epoch, 'prec1_T': valT_top1.item(), 'prec1_P': valP_top1.item(), 'loss': val_losses})
            if args.distributed:
                dist.barrier()

            # for param in model_without_ddp.parameters():
            #     param.grad = None

            # loss.backward()

            # # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(encode_model.parameters(), args.grad_clip)

            # optimizer.step()

            scheduler_en_t.step()
            print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopP@1: {:.4f}\t'.format(
                        epoch + 1, total_epochs, train_losses, trainT_top1, trainP_top1), flush=True)
                
            print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopP@1: {:.4f}\t'.format(
                        epoch + 1, total_epochs, val_losses, valT_top1, valP_top1),flush=True)

    # params_pi = list(privacy_image_model.parameters())
    # optimizer_pi = torch.optim.SGD(params_pi, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    # scheduler_pi = lr_scheduler.CosineAnnealingLR(optimizer_pi, T_max= total_epochs, eta_min=1e-7, verbose=True)
    
    # for epoch in range(0, total_epochs):
    #     print('\n Fixing Encoder Training Actor Recognition Now '+ str(epoch))

    #     '''
    #     Train actor model (privacy)
    #     '''

    #     trainT_top1, trainT_top5, train_losses = train(encode_model, privacy_image_model, train_loader, optimizer_pi, train_criterion, 'PI')

    #     train_logger_pi.log({'epoch': epoch,'prec1_T': trainT_top1.item(), 'loss': train_losses})

            
    #     valT_top1, valT_top5, val_losses = validate(encode_model, privacy_image_model, val_loader, optimizer_pi, train_criterion, 'PI')


    #     val_logger_pi.log({'epoch': epoch,'prec1_T': valT_top1.item(), 'loss': val_losses})

    #     scheduler_pi.step()

    #     print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}'.format(
    #                 epoch + 1, total_epochs, train_losses, trainT_top1), flush=True)
            
    #     print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\t'.format(epoch + 1, total_epochs, val_losses, valT_top1),flush=True)

def train_en(encode_model, action_recog_model, privacy_model, train_loader, optimizer, train_criterion, device):
    losses_degrad= AverageMeter()
    losses_target= AverageMeter()
    losses_budget= AverageMeter()
    losses= AverageMeter()

    top1_t= AverageMeter()
    top5_t= AverageMeter()

    top1_p= AverageMeter()
    top5_p= AverageMeter()

    encode_model.train()
    action_recog_model.train()
    privacy_model.eval()
    with tqdm(total=len(train_loader)) as t_bar:
        for k, (images, target_actor, target_action) in enumerate(train_loader):
            '''
            train encoder
            '''
            input = images.view(images.size()[0], -1, 3, 224, 224)
            batch_flow = []
            input = input.permute(0,2,1,3,4)
            for param in encode_model.parameters():
                param.grad = None


            # optimizer.zero_grad()
            # flows = []
            # for i in range(1, input.size()[2]):
            #     image1 = input[:,:,i-1,:,:].to(device)
            #     image2 = input[:,:,i,:,:].to(device)

            #     results_dict = encode_model(img0=image1, img1=image2,
            #                                 attn_splits_list=args.attn_splits_list,
            #                                 corr_radius_list=args.corr_radius_list,
            #                                 prop_radius_list=args.prop_radius_list,
            #                                 )
            #     flow_preds = results_dict['flow_preds']

            #     # print(flow_preds[0].size())
            #     flow = flow_preds[0].permute(0, 2, 3, 1).detach().cpu().numpy()  # [H, W, 2]
            #     flows.append(flow)


            # # print(input.size())
            for j in range(input.size()[0]):
                flows = []
                for i in range(1, input.size()[2]):
                    image1 = input[j,:,i-1,:,:].cuda(non_blocking=True)
                    image2 = input[j,:,i,:,:].cuda(non_blocking=True)

                    results_dict = encode_model(img0=image1, img1=image2,
                                                attn_splits_list=args.attn_splits_list,
                                                corr_radius_list=args.corr_radius_list,
                                                prop_radius_list=args.prop_radius_list,
                                                )
                    flow_preds = results_dict['flow_preds'][-1]

                    # print(flow_preds[0].size())
                    flow = flow_preds[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                    vis_flow = flow_to_image(flow)
                    # save_vis_flow_tofile(flow, 'test.png')
                    flows.append(flow)
                batch_flow.append(flows)
            # # print(len(batch_flow))
            # print(flows[0].shape)
            # batch_flow = np.array(flows)
            batch_flow = np.array(batch_flow)
            batch_flow = torch.from_numpy(batch_flow)
            batch_flow = batch_flow.to(device)

            '''
            train recognition
            '''
            # batch_flow = batch_flow.permute(1,4,0,2,3)
            batch_flow = batch_flow.permute(0,4,1,2,3)
            

            '''
            action
            '''
            output = action_recog_model(batch_flow.float())
            target_action = target_action.to(device)
            loss_target = train_criterion(output, target_action)
            prec1_target, prec5_target = accuracy(output, target_action)

            top1_t.update(prec1_target[0], images.size(0))
            top5_t.update(prec5_target[0], images.size(0))
            
            
            '''
            actor
            '''

            output = privacy_model(batch_flow.float())
            target_actor = target_actor.to(device)
            loss_actor = train_criterion(output, target_actor)

            prec1_actor, prec5_actor = accuracy(output, target_actor)

            top1_p.update(prec1_actor[0], images.size(0))
            top5_p.update(prec5_actor[0], images.size(0))


            loss = 2 * loss_target - loss_actor + 10.
            
            losses.update(loss.item(), images.size(0))

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(encode_model.parameters(), args.grad_clip)

            optimizer.step()
            
            t_bar.update(1)
    return top1_t.avg, top5_t.avg, top1_p.avg, top5_p.avg, losses.avg

def validate_en(encode_model, action_recog_model, privacy_model, train_loader, optimizer, train_criterion, device):
    losses_degrad= AverageMeter()
    losses_target= AverageMeter()
    losses_budget= AverageMeter()
    losses= AverageMeter()

    top1_t= AverageMeter()
    top5_t= AverageMeter()

    top1_p= AverageMeter()
    top5_p= AverageMeter()

    encode_model.eval()
    action_recog_model.eval()
    privacy_model.eval()

    with torch.no_grad(), tqdm(total=len(train_loader)) as t_bar:
        for k, (images, target_actor, target_action) in enumerate(train_loader):
            '''
            train encoder
            '''
            input = images.view(images.size()[0], -1, 3, 224, 224)
            batch_flow = []
            input = input.permute(0,2,1,3,4)


            flows = []
            # for i in range(1, input.size()[2]):
            #     image1 = input[:,:,i-1,:,:].to(device)
            #     image2 = input[:,:,i,:,:].to(device)

            #     results_dict = encode_model(img0=image1, img1=image2,
            #                                 attn_splits_list=args.attn_splits_list,
            #                                 corr_radius_list=args.corr_radius_list,
            #                                 prop_radius_list=args.prop_radius_list,
            #                                 )
            #     flow_preds = results_dict['flow_preds']

            #     # print(flow_preds[0].size())
            #     flow = flow_preds[0].permute(0, 2, 3, 1).detach().cpu().numpy()  # [H, W, 2]
            #     flows.append(flow)

            for j in range(input.size()[0]):
                flows = []
                for i in range(1, input.size()[2]):
                    image1 = input[j,:,i-1,:,:].cuda(non_blocking=True)
                    image2 = input[j,:,i,:,:].cuda(non_blocking=True)

                    results_dict = encode_model(img0=image1, img1=image2,
                                                attn_splits_list=args.attn_splits_list,
                                                corr_radius_list=args.corr_radius_list,
                                                prop_radius_list=args.prop_radius_list,
                                                )
                    flow_preds = results_dict['flow_preds'][-1]

                    # print(flow_preds[0].size())
                    flow = flow_preds[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                    vis_flow = flow_to_image(flow)
                    # save_vis_flow_tofile(flow, 'test.png')
                    flows.append(flow)
                batch_flow.append(flows)
            


            # batch_flow = np.array(flows)
            batch_flow = np.array(batch_flow)
            batch_flow = torch.from_numpy(batch_flow)
            batch_flow = batch_flow.to(device)

            '''
            train recognition
            '''
            # batch_flow = batch_flow.permute(1,4,0,2,3)
            batch_flow = batch_flow.permute(0,4,1,2,3)

            '''
            action
            '''
            output = action_recog_model(batch_flow.float())
            target_action = target_action.to(device)
            loss_target = train_criterion(output, target_action)
            prec1_target, prec5_target = accuracy(output, target_action)

            top1_t.update(prec1_target[0], images.size(0))
            top5_t.update(prec5_target[0], images.size(0))
            
            
            '''
            actor
            '''

            output = privacy_model(batch_flow.float())
            target_actor = target_actor.to(device)
            loss_actor = train_criterion(output, target_actor)
            prec1_actor, prec5_actor = accuracy(output, target_actor)

            top1_p.update(prec1_actor[0], images.size(0))
            top5_p.update(prec5_actor[0], images.size(0))


            loss = 2 * loss_target - loss_actor + 10.
            
            losses.update(loss.item(), images.size(0))


            t_bar.update(1)
    return top1_t.avg, top5_t.avg, top1_p.avg, top5_p.avg, losses.avg


def train(encode_model, recog_model, train_loader, optimizer, train_criterion, Type, device):
    encode_model.to(device)
    losses_degrad= AverageMeter()
    losses_target= AverageMeter()
    losses_budget= AverageMeter()
    losses= AverageMeter()

    top1= AverageMeter()
    top5= AverageMeter()

    encode_model.eval()
    recog_model.train()
    train_top1_acc = 0.0
    train_top5_acc = 0.0
    train_loss = 0.0

    with tqdm(total=len(train_loader)) as t_bar:
        for k, (images, target_actor, target_action) in enumerate(train_loader):
            '''
            train encoder
            '''
            input = images.view(images.size()[0], -1, 3, 224, 224)
            batch_flow = []
            batch_vis_flow = []
            input = input.permute(0,2,1,3,4)
            for j in range(input.size()[0]):
                flows = []
                vis_flows = []
                for i in range(1, input.size()[2]):
                    image1 = input[j,:,i-1,:,:].to(device)
                    image2 = input[j,:,i,:,:].to(device)

                    results_dict = encode_model(img0=image1, img1=image2,
                                                attn_splits_list=args.attn_splits_list,
                                                corr_radius_list=args.corr_radius_list,
                                                prop_radius_list=args.prop_radius_list,
                                                )
                    flow_preds = results_dict['flow_preds'][-1]

                    # print(flow_preds[0].size())
                    flow = flow_preds[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                    vis_flow = flow_to_image(flow)
                    # save_vis_flow_tofile(flow, 'test.png')
                    flows.append(flow)
                    vis_flows.append(vis_flow)
                    
                batch_flow.append(flows)
                batch_vis_flow.append(vis_flows)
            # print(len(batch_flow))
            batch_flow = np.array(batch_flow)
            batch_flow = torch.from_numpy(batch_flow)
            batch_flow = batch_flow.to(device)

            batch_vis_flow = np.array(batch_vis_flow)
            batch_vis_flow = torch.from_numpy(batch_vis_flow)
            batch_vis_flow = batch_vis_flow.to(device)

            '''
            train recognition
            '''
            
            batch_flow = batch_flow.permute(0,4,1,2,3)
            batch_vis_flow = batch_vis_flow.permute(0,4,1,2,3)

            # # # print(output_target.size())


            if Type == 'T':
                optimizer.zero_grad()
                output = recog_model(batch_flow.float())
                target_action = target_action.to(device)
                loss_target = train_criterion(output, target_action)
                
                loss_target.backward()
                optimizer.step()


                prec1_target, prec5_target = accuracy(output, target_action)

                train_top1_acc += prec1_target
                train_top5_acc += prec5_target
                train_loss += loss_target


                losses.update(loss_target.item(), images.size(0))

                top1.update(prec1_target[0], images.size(0))
                top5.update(prec5_target[0], images.size(0))
                

            elif Type == 'P':
                optimizer.zero_grad()
                output = recog_model(batch_flow.float())
                target_actor = target_actor.to(device)
                loss_actor = train_criterion(output, target_actor)
                
                loss_actor.backward()
                optimizer.step()


                prec1_actor, prec5_actor = accuracy(output, target_actor)

                train_top1_acc += prec1_actor
                train_top5_acc += prec5_actor
                train_loss += loss_actor


                losses.update(loss_actor.item(), images.size(0))

                top1.update(prec1_actor[0], images.size(0))
                top5.update(prec5_actor[0], images.size(0))

            elif Type == 'PI':
                output = []
                target_actor = target_actor.to(device)
                # print(batch_vis_flow.size())
                for r in range(batch_flow.size(2)):
                    output.append(recog_model(batch_vis_flow[:,:,r,:,:].float())) 
                loss_actor = 0
                for r in range(batch_flow.size(2)):            
                    loss_actor += train_criterion(output[r], target_actor)

                optimizer.zero_grad()
                loss_actor /= batch_flow.size(2) 
                loss_actor.backward()
                optimizer.step()

                prec1_actor= 0
                prec5_actor= 0
                for r in range(batch_flow.size(2)):            
                    prec1_actor_, prec5_actor_ = accuracy(output[r], target_actor)
                    prec1_actor += prec1_actor_
                    prec5_actor += prec5_actor_


                # prec1_actor, prec5_actor = accuracy(output, target_actor)
                losses.update(loss_actor.item(), images.size(0))

                top1.update(prec1_actor[0], images.size(0))
                top5.update(prec5_actor[0], images.size(0))
            
            t_bar.update(1)
    train_top1_sum = torch.tensor([train_top1_acc]).to(device)
    train_top5_sum = torch.tensor([train_top5_acc]).to(device)
    train_loss_sum = torch.tensor([train_loss]).to(device)

    # train_top1_avg = torch.distributed.all_reduce(train_top1_sum, op=torch.distributed.ReduceOp.SUM) / torch.distributed.get_world_size()
    # train_top5_avg = torch.distributed.all_reduce(train_top5_sum, op=torch.distributed.ReduceOp.SUM) / torch.distributed.get_world_size()
    # train_loss_avg = torch.distributed.all_reduce(train_loss_sum, op=torch.distributed.ReduceOp.SUM) / torch.distributed.get_world_size()

    train_loss_avg = train_loss_sum.item() / (images.size()[0] * len(train_loader))
    train_top1_avg = train_top1_sum.item() / (images.size()[0] * len(train_loader))
    train_top5_avg = train_top5_sum.item() / (images.size()[0] * len(train_loader))

    # return train_top1_avg, train_top5_avg, train_loss_avg
    return top1.avg, top5.avg, losses.avg

def validate(encode_model, recog_model, train_loader, optimizer, train_criterion, type, device):
    encode_model.to(device)
    losses_degrad= AverageMeter()
    losses_target= AverageMeter()
    losses_budget= AverageMeter()
    losses= AverageMeter()

    top1= AverageMeter()
    top5= AverageMeter()


    train_top1_acc = 0.0
    train_top5_acc = 0.0
    train_loss = 0.0

    encode_model.eval()
    recog_model.eval()
    with torch.no_grad(), tqdm(total=len(train_loader)) as t_bar:
        for k, (images, target_actor, target_action) in enumerate(train_loader):
            '''
            train encoder
            '''
            # print(images.size())
            input = images.view(images.size()[0], -1, 3, 224, 224)
            # print(input.size())
            batch_flow = []
            batch_vis_flow = []
            input = input.permute(0,2,1,3,4)
            for j in range(input.size()[0]):
                flows = []
                vis_flows = []
                for i in range(1, input.size()[2]):
                    image1 = input[j,:,i-1,:,:].to(device)
                    image2 = input[j,:,i,:,:].to(device)
                    # print(image1.size())
                    results_dict = encode_model(img0=image1, img1=image2,
                                attn_splits_list=args.attn_splits_list,
                                corr_radius_list=args.corr_radius_list,
                                prop_radius_list=args.prop_radius_list,
                                )
                    flow_preds = results_dict['flow_preds'][-1]
                    # print(flow_preds[0].size())
                    flow = flow_preds[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                    vis_flow = flow_to_image(flow)
                    # save_vis_flow_tofile(flow, 'test.png')
                    flows.append(flow)
                    vis_flows.append(vis_flow)

                batch_flow.append(flows)
                batch_vis_flow.append(vis_flows)
            # print(len(batch_flow))
            batch_flow = np.array(batch_flow)
            batch_flow = torch.from_numpy(batch_flow)
            batch_flow = batch_flow.to(device)

            batch_vis_flow = np.array(batch_vis_flow)
            batch_vis_flow = torch.from_numpy(batch_vis_flow)
            batch_vis_flow = batch_vis_flow.to(device)

            '''
            train recognition
            '''
            batch_flow = batch_flow.permute(0,4,1,2,3)
            batch_vis_flow = batch_vis_flow.permute(0,4,1,2,3)
            
    
            if type == 'T':
                output = recog_model(batch_flow.float())
                target_action = target_action.to(device)
                loss_target = train_criterion(output, target_action)
                prec1_target, prec5_target = accuracy(output, target_action)

                train_top1_acc += prec1_target
                train_top5_acc += prec5_target
                train_loss += loss_target

                # loss_target /= batch_flow.size(2) 
            
                losses.update(loss_target.item(), images.size(0))

                top1.update(prec1_target[0], images.size(0))
                top5.update(prec5_target[0], images.size(0))

            elif type == 'P':
                output = recog_model(batch_flow.float())
                target_actor = target_actor.to(device)
                loss_actor = train_criterion(output, target_actor)
                prec1_actor, prec5_actor = accuracy(output, target_actor)

                train_top1_acc += prec1_actor
                train_top5_acc += prec5_actor
                train_loss += loss_actor
                # loss_actor /= batch_flow.size(2) 


                losses.update(loss_actor.item(), images.size(0))

                top1.update(prec1_actor[0], images.size(0))
                top5.update(prec5_actor[0], images.size(0))

            elif type == 'PI':
                output = []
                target_actor = target_actor.to(device)
                # print(batch_vis_flow.size())
                for r in range(batch_flow.size(2)):
                    output.append(recog_model(batch_vis_flow[:,:,r,:,:].float())) 
                loss_actor = 0
                for r in range(batch_flow.size(2)):            
                    loss_actor += train_criterion(output[r], target_actor)
                loss_actor /= batch_flow.size(2) 

                prec1_actor= 0
                prec5_actor= 0
                for r in range(batch_flow.size(2)):            
                    prec1_actor_, prec5_actor_ = accuracy(output[r], target_actor)
                    prec1_actor += prec1_actor_
                    prec5_actor += prec5_actor_
                prec1_actor /= batch_flow.size(2) 
                prec5_actor /= batch_flow.size(2)  


                # prec1_actor, prec5_actor = accuracy(output, target_actor)
                losses.update(loss_actor.item(), images.size(0))

                top1.update(prec1_actor[0], images.size(0))
                top5.update(prec5_actor[0], images.size(0))

            t_bar.update(1)
    train_top1_sum = torch.tensor([train_top1_acc]).to(device)
    train_top5_sum = torch.tensor([train_top5_acc]).to(device)
    train_loss_sum = torch.tensor([train_loss]).to(device)
    # torch.distributed.all_reduce(train_top1_sum)
    # torch.distributed.all_reduce(train_top5_sum)
    # torch.distributed.all_reduce(train_loss_sum)

    train_loss_avg = train_loss_sum.item() / (images.size()[0] * len(train_loader))
    train_top1_avg = train_top1_sum.item() / (images.size()[0] * len(train_loader))
    train_top5_avg = train_top5_sum.item() / (images.size()[0] * len(train_loader))
    # return train_top1_avg, train_top5_avg, train_loss_avg

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main()