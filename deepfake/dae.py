
"""TensorFlow 2.0 implementation of vanilla Autoencoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import tensorflow as tf
from datetime import datetime

np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 10
learning_rate = 1e-2
intermediate_dim = 128
original_dim = 375

training_features = data.astype(np.float32)  # from mp4

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

logdir = "C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\dae_" + datetime.now().strftime("%Y%m%d-%H%M%S")

writer = tf.summary.create_file_writer(logdir)


with writer.as_default():
  with tf.summary.record_if(True):
    for epoch in range(epochs):
      for step, batch_features in enumerate(training_dataset):
        train(loss, autoencoder, opt, batch_features)
      loss_values = loss(autoencoder, batch_features)
      original = tf.reshape(batch_features, (batch_features.shape[0], 15, 25, 1))
      reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], 15, 25, 1))
      tf.summary.scalar('loss', loss_values, step=epoch  + 1)
      tf.summary.image('original', original, max_outputs=10, step=epoch  + 1)
      tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=epoch  + 1)



