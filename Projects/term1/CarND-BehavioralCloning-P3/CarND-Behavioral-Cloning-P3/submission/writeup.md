# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior *[ Done ]*
* Build, a convolution neural network in Keras that predicts steering angles from images *[ Done ]*
* Train and validate the model with a training and validation set *[ Done ]*
* Test that the model successfully drives around track one without leaving the road *[ Done ]*
* Summarize the results with a written report *[ Done ]*


[//]: # (Image References)

[image_center_driving]: ./images/good_driving.jpg "center lane driving"
[image_recovery1]: ./images/recovery.jpg "recover from left side"
[image_recovery2]: ./images/recovery2.jpg "recover from right side"
[image_curverun1]: ./images/curverun1.jpg "multiple curve run recordings"
[image_curverun2]: ./images/curverun2.jpg "multiple curve run recordings"
[image_reverse]: ./images/reverse.jpg "reversed driving"
[image_totalline]: ./images/Selection_005.png "total data points"

[image_model1]: ./images/model_basic.png "BasicNet model architecture"
[image_model2]: ./images/model_LeNet.png "LeNet model architecture"
[image_model3]: ./images/model_DeeperNet.png "DNet model architecture"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2a]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `Makefile` containing all related commands from training to running simulation
* `model.py` containing the main entry point for entire training
* `net.py` containing the script to build network, train, and save, etc
* `data.py` containing the script to process data (load data from disk, preprocess data, and generator module for training, etc)
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
make run.model.x
```

#### 3. Submission code is usable and readable

The `model.py` file contains the entry point ( `run()` ) towards model training. Inside, Network class object called __`BCNet()`__ and Data class object called __`BCData()`__ are created for data processing, model architecture creation, training, saving, etc. The entire codes are meant to be very simple for the best readability.
```python
# 1. create BCData class object
bcData = BCData(_ds, _fitgen=_data_fitgen)
# 2. create Behavioral Cloning Network class object with bcData
bcNet = BCNet(bcData)
# 3. build network
bcNet.buildNet(_nt)
# 4. train
bcNet.train(_n_epoch=_n_epoch)
# 5. save
bcNet.saveModel(_savef)

```
The detailed pipeline for data processing is in `data.py`, and for network training/validation is in `net.py` with comments.

### Model Architecture and Training Strategy

#### 1. Model architectures

First of all, I followed the same network architectures from the class. Then, I extended the LeNet architecture a bit deeper. Whereas the LeNet consists of 2 convolution layers with 5x5 kernel, 2 max pooling layers, 3 fully connected layers followed by a regression layer, the extended network architecture contains more number of convolutional layers with various kernel sizes. I kept the nonlinearity layer as same, the `ReLU`. Before the main body of the network, there's also a layer for data preprocessing, where I experimented various methods including cropping, normalization, and mean-shifting. I utilized Keras lamda layer, and also the fit_generator.

```python
 91 ##------------------------------
 92 # DEF basicNet()
 93 # - very basic net     
 94 #-------------------------------
 95 def basicNet(self):

100 ##------------------------------
101 # DEF LeNet()
102 # - LeNet
103 #   (http://yann.lecun.com/exdb/lenet/)
104 #-------------------------------
105 def LeNet(self):

116 ##------------------------------
117 # DEF dNet()
118 # - a bit deeper net
119 #-------------------------------
120 def dNet(self):

```

Here is a visualization of all the architecture mentioned above.

![alt text][image_model1]
![alt text][image_model2]
![alt text][image_model3]



#### 2. Attempts to reduce overfitting in the model

One approach I took is to add a regularizer such as dropout. Another is data augmentation (flipping the image, etc). The effectiveness of such approaches are verified by running the simulator with/without them and check if the vechicle stays on the track.

```python
# in net.py
128 self.model.add(Dense(100, activation='relu')
129 self.model.add(Dropout(0.5))    
130 self.model.add(Dense(50, activation='relu')
131 self.model.add(Dropout(0.5))    

# in data.py
 98 # augment data
 99 if self.augmented:
100 	images.append(np.fliplr(center_image))
101     angles.append(center_angle * -1.0)



```

#### 3. Model parameter tuning

I followed the same optimization method from the class, which is `adam` optimizer where the learning rate is adjusted automatically. Also, I used `mean_squared_error` for loss.

```python
# in net.py
134     ##------------------------------
135     # DEF opt()
136     # - set net loss and optimizer
137     #-------------------------------
138     def opt(self):
139         self.model.compile(loss='mse', optimizer='adam')
```


#### 4. Training data

I recorded many laps on track one using center lane driving. It required me a couple of practice run before I collect the good data.
Here is an example image of center lane driving:

![alt text][image_center_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover.

![alt text][image_recovery1]
![alt text][image_recovery2]

Also, I figured out during the data collection that, there are more cases that the steering is set to move straight, rather than moving left/right. I thought might result in skewed data point distribution in steering angles. Thus, after collecting 2-3 laps of normal driving, I collected additional data only recording in curved path (both right/left curves) for several times.

![alt text][image_curverun1]
![alt text][image_curverun2]


As mentioned in previous sections, I also flipped images and angles to make data have as much conditions as possible. Also, I recorded some portion of the lap by going backward.

![alt text][image_reverse]

The total number of data points I obtained from simulator is around `28,000`, by doing 5 recordings where each recording contains at least 2-3 full laps.

![alt text][image_totalline]


With the collected data, I did preprocessing such as shuffling, cropping, mean-shifting, flipping, etc.
```python
# in data.py

86 self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(self.height, self.width, self.channel)))

88 # preprocess data, centered around zero with small std 
   # (this or line 86, one way or another)
89 #model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(self.height, self.width, self.channel), 
90 #                                           output_shape=(self.height, self.width, self.channel)))

91 # crop images 60 rows pixels from the top of the image, 
92 #             25 rows pixels from the bottom of the image,
93 #             0 columns of pixels from the left of the image,
94 #         and 0 columns of pixels from the right of the image
95 model.add(Cropping2D(cropping=((60, 25), (0, 0))))


```


Then, I used 20% of it as validation samples.

```python
# in data.py
# 80% train samples, 20% validation samples
68 self.train_samples, self.validation_samples = train_test_split(samples, test_size=0.2)
```

#### 5. Training process

Training can be easily run in my framework, by using Makefile as follows:
```bash
$ make train.x 

```

In training, I first tried training with the provided data, using `basicNet()`, and I set `n_epoch=5`.

```python
6428/6428 [==============================] - 3s - loss: 1.3866 - val_loss: 1.2221
Epoch 2/5
6428/6428 [==============================] - 2s - loss: 5.3453 - val_loss: 3.8194
Epoch 3/5
6428/6428 [==============================] - 2s - loss: 4.8768 - val_loss: 2.6657
Epoch 4/5
6428/6428 [==============================] - 2s - loss: 3.0874 - val_loss: 4.4217
Epoch 5/5
6428/6428 [==============================] - 2s - loss: 3.2440 - val_loss: 3.5337
Saving model...
Model is saved to model_basic.h5!
```

The train and validation loss jumped a bit back and forth. Then, I replaced `basicNet()` with `LeNet()`, and kept `n_epoch=5`.

```python
6428/6428 [==============================] - 10s - loss: 1.8323 - val_loss: 0.0142
Epoch 2/5
6428/6428 [==============================] - 9s - loss: 0.0128 - val_loss: 0.0128
Epoch 3/5
6428/6428 [==============================] - 9s - loss: 0.0110 - val_loss: 0.0119
Epoch 4/5
6428/6428 [==============================] - 9s - loss: 0.0099 - val_loss: 0.0115
Epoch 5/5
6428/6428 [==============================] - 9s - loss: 0.0090 - val_loss: 0.0112
Saving model...
Model is saved to model_LeNet.h5!
```

The loss of both train and validation got a lot smaller than the `basicNet()`. However, I couldn't rule out a possibility of overfitting in this case, as I used small set of data (provided one). Thus, I moved on to larger dataset I already collected as explained in previous section, and also added regularization technique like `dropout` followed by fully connectied layers on `LeNet()`. I set `n_epoch` between 20 and 30 this time as my training data got a lot bigger than I planned.

```python
Train on 22724 samples, validate on 5682 samples
Epoch 1/30
22724/22724 [==============================] - 37s - loss: 0.5344 - val_loss: 0.0108
Epoch 2/30
22724/22724 [==============================] - 35s - loss: 0.0164 - val_loss: 0.0078
Epoch 3/30
22724/22724 [==============================] - 36s - loss: 0.0130 - val_loss: 0.0063
Epoch 4/30
22724/22724 [==============================] - 36s - loss: 0.0113 - val_loss: 0.0059
Epoch 5/30
22724/22724 [==============================] - 35s - loss: 0.0103 - val_loss: 0.0060
Epoch 6/30
22724/22724 [==============================] - 35s - loss: 0.0095 - val_loss: 0.0060
Epoch 7/30
22724/22724 [==============================] - 36s - loss: 0.0088 - val_loss: 0.0064
Epoch 8/30
22724/22724 [==============================] - 36s - loss: 0.0083 - val_loss: 0.0072
Epoch 9/30
22724/22724 [==============================] - 35s - loss: 0.0078 - val_loss: 0.0069
Epoch 10/30
22724/22724 [==============================] - 35s - loss: 0.0077 - val_loss: 0.0069
.
.
.
Epoch 28/30
22724/22724 [==============================] - 35s - loss: 0.0047 - val_loss: 0.0074
Epoch 29/30
22724/22724 [==============================] - 35s - loss: 0.0048 - val_loss: 0.0073
Epoch 30/30
22724/22724 [==============================] - 35s - loss: 0.0045 - val_loss: 0.0072
Saving model...
Model is saved to model_LeNet_all.h5!
```



Then I tried using a deeper network with also `dropout` layers. Also, as my network architecture is a bit big, and increasing the value of `n_epoch` helped for improving performance. Also, I incorporated fit_generator() to batch data processing. The following shows a sample train run result.



