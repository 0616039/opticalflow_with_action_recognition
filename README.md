# Privacy preserving action recognition

## Goal
With cameras becoming ubiquitous, concerns over privacy are becoming increasingly important. As such, we aim to create a system that can safeguard privacy without compromising the accuracy of action recognition results.

## Related work

[1] GMFlow: Learning Optical Flow via Global Matching, CVPR 2022
![](https://github.com/0616039/opticalflow_with_action_recognition/blob/main/GMFlow.png)

[2] Privacy-Preserving Action Recognition via Motion Difference Quantization, ECCV 2022
![](https://github.com/0616039/opticalflow_with_action_recognition/blob/main/BDQ.png)

## The propose network

Use optical flow method (GMFlow [1]) as encoder, 3D ResNet as Target network and Privacy network . Then, we train the network like GAN as BDQ encoder [2], that is, our objective is to decrease the accuracy of person recognition while maintaining the accuracy of action recognition. 


![](https://github.com/0616039/opticalflow_with_action_recognition/blob/main/MyNetwork.png)

## Results
The results of [2] is on the left side, the action recognition is about 83%, while the actor recognition is 35%.

Our experiment results is on the right side, as we can see, on the first row, if we just use the pretrained weight of the GMFlow[1], we get same score on the action recognition, but get lower accuracy on actor recognition.

Next, we experimented with different training approaches in the second and third rows. We trained the Target network and Privacy network in the first and second steps, and then varied the third step. In the second row, we only trained the encoder, while in the third row, we trained both the encoder and the Target network together.

Ultimately, we found that training the encoder with the Target network produced the best results.
![](https://github.com/0616039/opticalflow_with_action_recognition/blob/main/Result.png)
