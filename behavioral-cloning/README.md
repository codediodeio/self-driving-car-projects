# Self-Driving Car Behavioral Cloning in Keras

A deep learning model for predicting steering angles using raw image frames from the Udacity driving simulator.

# Data Collection and Processing

Data was collected by driving 2 laps on the track without veering outside of the lines, followed by shorter recovery segments to simulate what the car should do when it heads off track. Input data consisted of shuffled image frames at a size of 120x320x3 pixels. The upper 40 y-axis pixes were cropped off, as they only added extra noise to the data. The final model was trained on 3141 samples, with 20% used for validation.  

![training image](https://storage.googleapis.com/kaggle-data/center_2016_12_10_13_38_36_460.jpg)


# Model Architecture

The model was loosely based on Nvidia's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper.

The model starts with a BatchNormalization layer that sets the mean activation to 0 and the standard deviation to 1. This step ensures that pixel values each batch of images are standardized scaled.  

Four convolutional layers are used for feature extraction, each with a 3x3 kernel size. Each convolutional layer is followed by max pooling with 2x2 strides. The convolutional layers are flattened and run through two fully connected layers. A 0.25 dropout layer was added after the final ReLU activation, meaning 25% of the input units have a chance of being dropped to prevent overfitting. The output layer uses a linear activation to make steering angle predictions as a continuous value ranging from -1.0 to 1.0

Layer             (Batch, Rows, Cols, Channels)
====================================================================================================
(BatchNormalize)  (None, 120, 320, 3)  
(Convolution2D)   (None, 118, 318, 3)    
(MaxPooling2D)    (None, 59, 159, 3)    
Convolution2D)    (None, 57, 157, 9)         
(MaxPooling2D)    (None, 28, 78, 9)          
(Convolution2D)   (None, 26, 76, 18)           
(MaxPooling2D)    (None, 13, 38, 18)        
(Convolution2D)   (None, 11, 36, 32)          
(MaxPooling2D)    (None, 5, 18, 32)       
(Flatten)         (None, 2880)       
(Dense)           (None, 80)           

(Dense)           (None, 15)              
(Dropout)         (None, 15)             
(Dense)           (None, 1)              
====================================================================================================
Total params: 238979


# Hyperparameters

- Learning Rate: 0.0001
- Epochs: 40
- Batch Size: 64
- Optimizer: RMSprop
- Cost: Mean Squared Error

Learning rate, epochs, and batch size were determined through trial and error. Overfitting usually started around 30 epochs, so the `ModelCheckpoint` and  `EarlyStopping` callbacks in Keras were used to save the best weights and stop training when the validation loss stopped decreasing.

RMSprop works by dividing	the	learning rate by moving average of recent	gradients	for a weight. I had originally intended to use the Adam optimizer, which is similar to RMSprop, but also tracks the decay of past gradients. However, RMSprop consistently provided better results for this problem.

Mean squared error (MSE) proved to be the best loss function, in comparison root mean squared error (RMSE) and mean absolute error (MAE). The problem with RMSE is that is penalizes large errors, so you end up with a car that is afraid to take turns and eventually drives off the road. The problem with MAE is that its less sensitive to large errors, so you'll get an occasional sharp turn that throws off all subsequent actions.
