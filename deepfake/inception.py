

import numpy as np

from pathlib import Path
from matplotlib.image import imread
from matplotlib import pyplot as plt

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

input_dir = Path("C:\\Users\\T149900\\Downloads\\cats_and_dogs")

output_dir = Path("C:\\Users\\T149900\\Downloads\\cats_and_dogs_processed")

assert input_dir.is_dir()
assert output_dir.is_dir()

l_photos = list()
l_labels = list()


for x in list (input_dir.iterdir()):
    output = 0.0
    if x.name.startswith("cat"):
        output = 1.0
    
    photo = load_img(x, target_size=(200, 200))    
    photo = img_to_array(photo)

    l_photos.append(photo)
    l_labels.append(output)
"""c"""


photos = np.asarray(l_photos)
labels = np.asarray(l_labels)
print(photos.shape, labels.shape)

np.save(output_dir / 'dogs_vs_cats_photos.npy', photos)
np.save(output_dir / 'dogs_vs_cats_labels.npy', labels)

##################################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from pathlib import Path


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


from keras.preprocessing.image import ImageDataGenerator


input_dir = Path("C:\\Users\\T149900\\Downloads\\cats_and_dogs_processed")

photos = np.load(input_dir / 'dogs_vs_cats_photos.npy')
labels = np.load(input_dir / 'dogs_vs_cats_labels.npy')

print(photos.shape, labels.shape)

def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

model = define_model()


# CONTINUE HERE



photos = photos / 255.0

num_all = photos.shape[0]
num_test = 5000
num_train = num_all - num_test

idx = np.array(range(num_all))
idx_test = np.random.choice(idx, size = num_test, replace = False)
idx_train = np.array(list(set(idx)  - set(idx_test)))

anTrain = photos[idx_train]
anTest  = photos[idx_test]

anYTrain = labels[idx_train]
anYTest  = labels[idx_test]



history = model.fit(x = anTrain, y = anYTrain, batch_size = 64, epochs = 20, verbose = 1, validation_data=[anTest, anYTest])