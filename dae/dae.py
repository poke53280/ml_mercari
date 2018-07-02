
# Building Autoencoders in Keras
# https://blog.keras.io/building-autoencoders-in-keras.html
#
#
#
#
# https://hsaghir.github.io/data_science/denoising-vs-variational-autoencoder/
#
# Keras example of variational auto encoder
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
#
#


from keras.layers import Input, Dense
from keras.models import Model


# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats


# this is our input placeholder
input_img = Input(shape=(784,))


# The encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

autoencoder.summary()



# Access to encode model
encoder = Model(input_img, encoded)



# As well as the decoder model:

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]


# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)


# Now let's train our autoencoder for 50 epochs:

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))


# encode the test digits

encoded_imgs = encoder.predict(x_test)


# ... and decode them
decoded_imgs = decoder.predict(encoded_imgs)


import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#
# Michael Jahrer:
# Seguro Winning Solution
# Presentation
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
#
# 'Imperfect implementation'
# https://www.kaggle.com/osciiart/denoising-autoencoder
#
# Detailed discussion
# http://forums.fast.ai/t/porto-seguro-winning-solution-representation-learning/8499/32
#
#
# 
# Why Does Unsupervised Pre-training Help Deep Learning?
# http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf
#
#

#############################################################################
#
#
# Sparse to dictionary 

import numpy as np
import pandas as pd

_interval_index = 0
_wait_indexer = 0

def my_func(x):
    an = np.array(x)    
    m = (an != 0)

    nz = np.where(m)

    dict =  {}

    for x in nz[0]:
        value = an[x]
        assert (value > 0)

        value = _interval_index.get_loc(np.log(value))

        dict[x] = value

    return dict

def wait_find(d):
    k = list (d.keys())
    vdiff = np.diff(k)
    v = [k[0]]
    v.extend(vdiff)
    
    return dict(zip(v, d.values()))


def convert_wait(d):
    d2 = {}
    
    for k, v in d.items():
        wait_time = k
        amount = v

        time_cat = _wait_indexer.get_loc(wait_time)

        d2[time_cat] = amount

    return d2            
            
"""c"""      

def categorize_transactions(train, count):

    X_train = np.array(train)
    X_train = X_train.flatten()

    m = (X_train != 0)

    X_train = X_train[m]

    X_train = np.log(X_train)

    s = pd.qcut(X_train, count, duplicates='drop')

    s.value_counts()

    interval_index = s._categories

    return interval_index

"""c"""


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'train.csv')

y_trainFull = np.log(train.target)
train_id = train.ID
train = train.drop(['target', 'ID'], axis = 1)

_interval_index = categorize_transactions (train, 50)

q = train.apply(my_func, raw = True, axis = 1)


q = q.apply(wait_find)

wait_periods = []

for x in q:
    wait_periods.extend(x)
"""c"""

anwait_period = np.array(wait_periods)


# Time categories:

s = pd.qcut(anwait_period, 10, duplicates='drop')

_wait_indexer = s._categories


w2 = q.apply(convert_wait)


def get_string(d):

    info = ""

    for (w, a) in d.items():
        wait = w
        value = a
        info = info + f"WAIT_{wait} AMT_{value} "

    return info
   
"""c"""

s = w2.apply(get_string)



#####################
#
#
# Re-arrange so most non-null column to the left

# Continue here


X = np.array(train)

X_nz = np.count_nonzero(train, axis = 0)

idx = np.argsort(X_nz)

X = X[idx]