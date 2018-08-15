
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
#
# http://kvfrans.com/variational-autoencoders-explained/
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
# Why Does Unsupervised Pre-training Help Deep Learning?
# http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf
#
#





