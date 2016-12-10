# Self-Driving Car Behavioral Cloning in Keras

A deep learning model for predicting steering angles using raw image frames from the Udacity driving simulator.

# Data Collection and Processing

Data was collected by driving 2 laps on the track without veering outside of the lines, followed by shorter recovery segments to simulate what the car should do when it heads off track. Input data consisted of shuffled image frames at a size of 160x320x3 pixels. The final model was trained on 1937 samples, and validated on 485 samples - an 80/20 split.

![training image](https://storage.googleapis.com/kaggle-data/center_2016_12_10_13_38_36_460.jpg)


# Model Architecture

The model was loosely based on Nvidia's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper.

Four convolutional layers are used for feature extraction, each with a 3x3 kernel size. Each convolutuonal layer is followed by max pooling with 2x2 strides. The convolutional layers are flattened and run through two fully connected layers. A 0.25 dropout layer was added after the final ReLU activation, meaning 25% of the input units have a chance of being dropped to prevent overfitting. The output layer uses a linear activation to make steering angle predictions as a continuous value ranging from -1.0 to 1.0

- Input:    (160, 320, 3)
- Conv 3x3: (160, 320, 3)
- Max Pool: (80, 160, 3)
- Conv 3x3: (80, 160, 24)
- Max Pool: (40, 80, 24)
- Conv 3x3: (40, 80, 36)
- Max Pool: (20, 40, 36)
- Conv 3x3: (20, 40, 48)
- Max Pool: (10, 20, 48)
- Flatten:  (9600)
- Dense:    (80)
- Dense:    (16)
- Dropout:  (0.25)
- Output:   (1)            

Total Params: 793561

# Hyperparameters

- Learning Rate: 0.0001
- Epochs: 40
- Batch Size: 64
- Optimizer: RMSprop
- Cost: Mean Squared Error

Learning rate, epochs, and batch size were determined through trial and error. Overfitting usually started around 30 epochs, so the `ModelCheckpoint` and  `EarlyStopping` callbacks in Keras were used to save the best weights and stop training when the validation loss stopped decreasing.

RMSprop works by dividing	the	learning rate by moving average of recent	gradients	for a weight. I had originally intended to use the Adam optimizer, which is similar to RMSprop, but also tracks the decay of past gradients. However, RMSprop consistently provided better results for this problem.

Mean squared error (MSE) proved to be the best loss function, in comparison root mean squared error (RMSE) and mean absolute error (MAE). The problem with RMSE is that is penalizes large errors, so you end up with a car that is afraid to take turns and eventually drives off the road. The problem with MAE is that its less sensitive to large errors, so you'll get an occasional sharp turn that throws off all subsequent actions.
