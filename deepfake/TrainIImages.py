

import numpy as np
from pathlib import Path
from mp4_frames import get_ready_data_dir

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD


def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 256, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


def define_model3():
    model = Sequential()
    model.add(Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 256, 3)))

    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_model_lstm():


    model = Sequential()
    model.add(LSTM(2048, activation='relu', return_sequences=True, input_shape=(32, 256*3)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
   
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def get_model_dense():
    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(32, 256, 3)))

    model.add(Flatten())

    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(1, activation='sigmoid'))
   
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model



ready_dir = get_ready_data_dir()

model = get_model_dense()

filepath_photo = ready_dir / f"photos_0001.npy"
filepath_label = ready_dir / f"labels_0001.npy"

anTest = np.load(filepath_photo)
#anTest = anTest.reshape(-1, 32, 256 * 3)

anYTest = np.load(filepath_label)

for x in range(1):
    print(f"Processing {x:04}...")
    filepath_photo = ready_dir / f"photos_{x:04}.npy"
    filepath_label = ready_dir / f"labels_{x:04}.npy"

    if filepath_photo.is_file() and filepath_label.is_file():
        pass
    else:
        print("Out of data")
        break

    anTrain = np.load(filepath_photo)
    anYTrain  = np.load(filepath_label)

    idx = np.arange(anTrain.shape[0])
    np.random.shuffle(idx)

    anTrain = anTrain[idx]
    anYTrain = anYTrain[idx]
    #anTrain = anTrain.reshape(-1, 32, 256 * 3)


    history = model.fit(x = anTrain, y = anYTrain, batch_size = 128, epochs = 20, verbose = 1, validation_data=[anTest, anYTest])




