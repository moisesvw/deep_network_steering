# End to End DeepNetwork implementatino for Driving Steering

This projects I built a deep learning system that learns to steer car wheels in a Simulator.

I've created a shell script that synchronizes the network and the data in a remote server, and then run the training, test the resulting model and finally pulls the model to the local machine. I used amazon p3.2xlarge instance.

You can download the [Simulator](https://github.com/udacity/self-driving-car-sim/releases) from Udacity Repo to collect data for training and test the model. For more information, please refer to the Udacity Self-Driving Car Nanodegree [repo](https://github.com/udacity/CarND-Behavioral-Cloning-P3). Once you obtain data, this project assumes you put the data in the directory /data/train and /data/test.

To run the training execute this command `./upload_and_execute`, this will run the network on the remote server, feel free to change the host `p` for your host or IP address host.

Results of training are shown in the next video

[![Results](https://img.youtube.com/vi/87LeecbK604/0.jpg)](https://www.youtube.com/watch?v=87LeecbK604)

 --

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/placeholder.png "Model Visualization"
[image2]: ./images/center_2017_04_22_11_28_57_268.jpg "Center"
[image3]: ./images/center_2017_04_22_11_28_44_299.jpg "Recovery Image"
[image4]: ./images/center_2017_04_22_11_28_46_071.jpg "Recovery Image"
[image5]: ./images/center_2017_04_22_11_28_47_600.jpg "Recovery Image"
[image6]: ./images/imorig.png "Normal Image"
[image7]: ./images/imflip.png "Flipped Image"
[image8]: ./images/track1dist.png "Data track one"
[image9]: ./images/train2dist.png "Data track two"
[image10]: ./images/track2ddist.png "Data track two"


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* worked_models a folder with other experiments
* train2, track2d, track1 folders containing training data
* lab.ipynb notebook for experiments

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on [NVIDIA end to end steering paper](https://arxiv.org/abs/1604.07316), the configuration of the network it as below (model.py lines 122-142):

```
| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3 RGB->YUV image                      | 
| Cropping Layer        | cut off 25px top, 25px bottom, out 70x320x3   |
| Lambda Layer          | normalization                                 |
| Convolution 5x5x24    | 2x2 stride, 'VALID' padding, outputs 33x158x24|
| RELU                  |                                               |
| Convolution 5x5x36    | 2x2 stride, 'VALID' padding, outputs 15x77x36 |
| RELU                  |                                               |
| Convolution 5x5x48    | 2x2 stride, 'VALID' padding, outputs 6x37x48  |
| RELU                  |                                               |
| Convolution 3x3x64    | 1x1 stride, 'VALID' padding, outputs 4x35x64  |
| RELU                  |                                               |
| Convolution 3x3x64    | 1x1 stride, 'VALID' padding, outputs 2x33x64  |
| RELU                  |                                               |
| Flatten               |                                               |
| DROPOUT               | keep probability 0.50                         | 
| Fully connected       | input 4224                                    |
| Fully connected       | input 100                                     |
| Fully connected       | input 50                                      |
| Fully connected       | input 10                                      |
| Fully connected       | input 1                                       |


```

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 136). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 74-78) (code lines 107-108).

The model was tested on a test set and also by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, and the initial learning rate was `0.0006` (model.py line 71). Batch size for generate data was 64. I use between 3 and 15 epochs.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I ran roughly 120 experiments starting with a basic convnet just one convolution, one flatten and one fully connected layer of size 1, then I've used the model base on [NVIDIA end to end steering paper](https://arxiv.org/abs/1604.07316).

Start with base line model and record in track one using keyboard, it gets about 50 loss score and car drove insane in track one.

By augmenting data, validation and test scores were 0.089 and 0.092, car start to drive better but keep hard driving to the right. Next thing to do is cropping image to get rid off the most top part of the image which is forest and part of the landscape it may helps to squeeze some noise. Doing that get better results on validation and test sets, the car did slightly better but this time with a hard steering to the left.

The next step it is augment network complexity by adding two more fully connected layers and one more convolution.

Then I use the NVDIA based model and normalize data. It improves the loss but car still drove in circles some times, next step was augment data with all cameras and flip images.

I used the data provided by Udacity course first but I did not get good results, then I have tried by recording multiple data sets from track two, I was thinking I could train well on track two and get good results on both track one and track two, it didn't work well, some times it work well on track one but not track two and vice versa.

Despite of getting very good scores without indication of overfitting the car did not drive well and get off the lane in some point of the track. 

The overall strategy for deriving a model architecture was to run multiple experiments changing number of epochs, correction angle for left and right camera
images, adding multiple dropout layers between fully connected layers, combine data,
train with new data recorded using a gamer mouse, increasing decreasing initial learning rate and bath_size for the data generators.

Experiment:
* `worked_models/nvda_4_52` trained in track two  works almost perfect in track one, but bad in track two.
* `worked_models/nvda_4_57` trained in track two work almost well in track two and part of track one.
* `worked_models/nvda_6_3`  trained in track two  works almost perfest in track two, but kind of good in track one.



for the track two I have trained a the same model with different parameters that works well once, I've tried to reproduce the experiment but I did not get the car complete the track two with new training, I suppose that the generators did not feed model with the same data as experiment `worked_models/nvda_6_3`.

This was the output of experiment nvda_6_3 which make the car drive in track two:
Adam learning rate: 0.0006
Steering correction for left and right cameras 0.22
generator_batch 64

datasets used for training `train2` and `track2d`
```
541/540 [==============================] - 132s - loss: 0.0445 - validation_loss: 0.0313
Epoch 2/4
541/540 [==============================] - 129s - loss: 0.0292 - validation_loss: 0.0291
Epoch 3/4
541/540 [==============================] - 129s - loss: 0.0247 - validation_loss: 0.0255
Epoch 4/4
541/540 [==============================] - 129s - loss: 0.0213 - validation_loss: 0.0193

Test loss 0.0205293367707
```

For track one, I re-record data this time with a mouse, I did one lap in both directions, and also
re record in some curves specially in the curve to take the shortcut.

when augmenting data, I added a random condition of 0.5 to add flipped images. And a random
condition of 0.7 to add left and right recover images. The last one was key to get the car
drive more smother.

Final loss scores were train loss: 0.0075 - validation loss: 0.0083 - test loss 0.0077

At the end of the process, the vehicle is able to drive autonomously around the track one without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 122-141) consisted of a convolution neural network based on [NVIDIA end to end steering paper](https://arxiv.org/abs/1604.07316), with the following layers and layer sizes:
```
| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3 RGB->YUV image                      | 
| Cropping Layer        | cut off 25px top, 25px bottom, out 70x320x3   |
| Lambda Layer          | normalization                                 |
| Convolution 5x5x24    | 2x2 stride, 'VALID' padding, outputs 33x158x24|
| RELU                  |                                               |
| Convolution 5x5x36    | 2x2 stride, 'VALID' padding, outputs 15x77x36 |
| RELU                  |                                               |
| Convolution 5x5x48    | 2x2 stride, 'VALID' padding, outputs 6x37x48  |
| RELU                  |                                               |
| Convolution 3x3x64    | 1x1 stride, 'VALID' padding, outputs 4x35x64  |
| RELU                  |                                               |
| Convolution 3x3x64    | 1x1 stride, 'VALID' padding, outputs 2x33x64  |
| RELU                  |                                               |
| Flatten               |                                               |
| DROPOUT               | keep probability 0.50                         | 
| Fully connected       | input 4224                                    |
| Fully connected       | input 100                                     |
| Fully connected       | input 50                                      |
| Fully connected       | input 10                                      |
| Fully connected       | input 1                                       |
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded for track one one lap in both directions using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the left or right sides, this images shows how to recover from right.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would good for creating more case the model could learn from drive, For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


Basic stats for track one data
```
Total 3083
Positive angles count 1220
Negative angles  count 1822
Neutral count 41
Max: 0.3186173
Min: -0.3212338
Mean: -0.0117456327135
Median: -0.008844339
Mode: 0    0.017394
```
The distribution of data for track one looks like this:

![alt text][image8]



Basic stats for track two train2 data
```
Total 2943
Positive angles count 1406
Negative angles count 1505
Neutral count 32
Max: 0.9285452
Min: -0.7305424
Mean: -0.00857330559524
Median: -0.006375295
Mode: 0    0.0
```
The distribution of data for track two train2 data looks like this:

![alt text][image9]


Basic stats for track two track2d data
```
Total 4263
Positive angles count 1981
Negative angles count 2221
Neutral count 61
Max: 1.0
Min: -0.889388
Mean: 0.00335523906772
Median: -0.01445313
Mode: 0    0.0
```
The distribution of data for track two track2d data looks like this:

![alt text][image10]

Finally the data was feed by generators to avoid load entire data in memory,
also the flip augmentation was perform in feed.
