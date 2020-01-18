
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import tensorflow as tf
from datetime import datetime
from skimage.draw import line


# https://machinelearningmastery.com/lstm-autoencoders/

np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 10
learning_rate = 1e-2
intermediate_dim = 128
original_dim = 784

(training_features, _), _ = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)
training_features = training_features.reshape(training_features.shape[0],
                                              training_features.shape[1] * training_features.shape[2])
training_features = training_features.astype('float32')

training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
training_dataset = training_dataset.batch(batch_size)
training_dataset = training_dataset.shuffle(training_features.shape[0])
training_dataset = training_dataset.prefetch(batch_size * 4)


class Encoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim):
    super(Encoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu,
      kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.sigmoid
    )
    
  def call(self, input_features):
    activation = self.hidden_layer(input_features)
    return self.output_layer(activation)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim):
    super(Decoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu,
      kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=original_dim,
      activation=tf.nn.sigmoid
    )
  
  def call(self, code):
    activation = self.hidden_layer(code)
    return self.output_layer(activation)
  
class Autoencoder(tf.keras.Model):
  def __init__(self, intermediate_dim, original_dim):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder(intermediate_dim=intermediate_dim)
    self.decoder = Decoder(
      intermediate_dim=intermediate_dim,
      original_dim=original_dim
    )
  
  def call(self, input_features):
    code = self.encoder(input_features)
    reconstructed = self.decoder(code)
    return reconstructed

autoencoder = Autoencoder(
  intermediate_dim=intermediate_dim,
  original_dim=original_dim
)
opt = tf.optimizers.Adam(learning_rate=learning_rate)

def loss(model, original):
  reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
  return reconstruction_error
  
def train(loss, model, opt, original):
  with tf.GradientTape() as tape:
    gradients = tape.gradient(loss(model, original), model.trainable_variables)
  gradient_variables = zip(gradients, model.trainable_variables)
  opt.apply_gradients(gradient_variables)


logdir = f"C:\\Users\\T149900\\source\\repos\\sm_data\\sm_data\\dae_{intermediate_dim}" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(logdir)

step = 0

with writer.as_default():
  with tf.summary.record_if(True):
    for epoch in range(epochs):
      for batch_features in training_dataset:
        train(loss, autoencoder, opt, batch_features)
        loss_values = loss(autoencoder, batch_features)
        original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
        reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], 28, 28, 1))
        tf.summary.scalar('loss', loss_values, step=step)
        tf.summary.image('original', original, max_outputs=4, step=step)
        tf.summary.image('reconstructed', reconstructed, max_outputs=4, step=step)
        step = step + 1



l_img = []

for ix, x in enumerate(training_features):
    print (ix)
    img = x.reshape(28, 28)
    acImg = get_lines_from_image(img, 2048)
    l_img.append(acImg)

afData = np.vstack(l_img)


np.save("C:\\Users\\T149900\\source\\repos\\sm_data\\sm_data\\data_16.npy", afData)


def get_line(img):

    length = 16

    x0 = 5 + np.random.choice(5)
    y0 = np.random.choice(28)

    x1 = x0 + np.random.choice(2) + length + 1
    y1 = np.random.choice(28)

    assert x1 < 28, "x1 < 28"
    

    rr, cc = line(x0, y0, x1, y1)

    data = img[rr, cc][:length]

    assert data.shape[0] == 16

    return data

def get_lines_from_image(img, data_max):

    samples = 0
    acData = np.zeros((data_max, 16), dtype = np.float32)

    x = 0
    while x < data_max:
        data = get_line(img)
        if not (data == 0).all():
            acData[x] = data
            x = x + 1

        samples = samples + 1

    return acData


