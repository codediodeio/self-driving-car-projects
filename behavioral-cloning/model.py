import os, cv2, random, json
import numpy as np
import pandas as pd
np.random.seed(23)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

ROWS = 120
COLS = 320
CHANNELS = 3
DIR = 'data/IMG/'

nb_epoch = 40
batch_size = 64

def img_id(path):
    return path.split('/IMG/')[1]

def read_image(path):
    """Read image and reverse channels"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img[40:160, 0:320] # Cropping top 40 y axis pixels
    return img[:,:,::-1]

def fit_gen(data, batch_size):
    """Python generator to read/process data on the fly"""
    while 1:
        ## Initilize empty batch data
        x = np.ndarray((batch_size, ROWS, COLS, CHANNELS), dtype=np.uint8)
        y = np.zeros(batch_size)
        i=0
        for line in data.iterrows():
            ## process (x, y) line in batch
            path = line[1].center.split('/IMG/')[1]
            x[i] = read_image(DIR+path)
            y[i] = line[1].angle
            i+=1
            if i == batch_size:
                ## When batch size is reached, yield data and reset variables
                i=0
                yield (x, y)
                x = np.ndarray((batch_size, ROWS, COLS, CHANNELS), dtype=np.uint8)
                y = np.zeros(batch_size)

def get_model():
    """Define hyperparameters and compile model"""

    lr = 0.0001
    weight_init='glorot_normal'
    opt = RMSprop(lr)
    loss = 'mean_squared_error'

    model = Sequential()

    model.add(BatchNormalization(mode=2, axis=1, input_shape=(ROWS, COLS, CHANNELS)))
    model.add(Convolution2D(3, 3, 3, init=weight_init, border_mode='valid', activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(9, 3, 3, init=weight_init, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(18, 3, 3, init=weight_init, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, init=weight_init, border_mode='valid',  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(80, activation='relu', init=weight_init))

    model.add(Dense(15, activation='relu', init=weight_init))

    model.add(Dropout(0.25))
    model.add(Dense(1, init=weight_init, activation='linear'))

    model.compile(optimizer=opt, loss=loss)

    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)


if __name__ == '__main__':

    print("Processing Data -------------")

    data = pd.read_csv('data/driving_log.csv', header=None,
                       names=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])

    y_all = data.angle.values
    n_samples = y_all.shape[0]
    print("Training Model with {} Samples".format(n_samples))

    image_paths = data.center.apply(img_id).values.tolist()


    ## Load all data (optional for validation only)
    X_all = np.ndarray((n_samples, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, path in enumerate(image_paths):
        DIR+path
        img = read_image(DIR+path)
        X_all[i] = img

    # Validation Split
    X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=23)

    print("Processing Complete ---------")

    print("Training Model --------------")

    model = get_model()

    model.fit_generator(fit_gen(data, batch_size),
        samples_per_epoch=data.shape[0], nb_epoch=nb_epoch,
        validation_data=(X_test, y_test), callbacks=[save_weights, early_stopping])

    ## uncomment when using all data in memory
    # model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #           validation_data=(X_test, y_test), verbose=1, shuffle=True, callbacks=[save_weights, early_stopping])


    preds = model.predict(X_test, verbose=1)

    print("Results ---------------------")

    print( "Test MSE: {}".format(mean_squared_error(y_test, preds)))
    print( "Test RMSE: {}".format(np.sqrt(mean_squared_error(y_test, preds))))

    # Save model to JSON
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    print("Finished! -------------------")
