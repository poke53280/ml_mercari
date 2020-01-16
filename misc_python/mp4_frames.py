

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import signal

import matplotlib.animation as animation

from mtcnn.mtcnn import MTCNN


####################################################################################
#
#   m_desc
#

def m_desc(m):
    if m.shape[0] == 0:
        return "EMPTY"

    elif m.shape[0] == m.sum():
        return f"ALL [{m.sum()}]"

    isAll = (m.sum() == m.shape[0])
    isNone = (m.sum() == 0)
    rPct = 100.0 * (m.sum() / m.shape[0])
    zPct = f"{rPct:.1f}"
    is100pct = f"{rPct:.1f}" == "100.0"
    is0pct = f"{rPct:.1f}" == "0.0"

    zDesc = ""

    if isAll:
        zDesc = "ALL"
    elif is100pct:
        zDesc = "<100%"
    elif isNone:
        zDesc = "NONE"
    elif is0pct:
        zDesc = ">0%"
    else:       
        zDesc = zPct + "%"

    zRes = f"{zDesc} [{m.sum()}]/[{m.shape[0]}]"

    return zRes
"""c"""






####################################################################################
#
#   detect_faces
#


def detect_faces(vidcap):
    
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = vidcap.get(cv2.CAP_PROP_FPS)

    nFrame = length
    iFrame = 0

    
    left_eye = []
    right_eye = []
    nose = []
    left_mouth = []
    right_mount = []
    confidence = []

    detector = MTCNN()

    l_image = []

    for iFrame in range (4):

        print(f"Processing {iFrame}...")

        success,image = vidcap.read()

        l_image.append(image)
        
        faces = detector.detect_faces(image)



        if len(faces) > 0:
            f = faces[0]
            left_eye.append(f['keypoints']['left_eye'])
            right_eye.append(f['keypoints']['right_eye'])
            nose.append(f['keypoints']['nose'])
            left_mouth.append(f['keypoints']['mouth_left'])
            right_mount.append(f['keypoints']['mouth_right'])
            confidence.append(f['confidence'])

        for x in faces:
            print (x['confidence'])


    df = pd.DataFrame({'confidence' : confidence, 'left_eye' : left_eye, 'right_eye' : right_eye, 'nose' : nose, 'left_mouth' : left_mouth, 'right_mouth' : right_mount})

    from sklearn.metrics import mean_squared_error

    # Area around point 4 x 4

    w = 8
    x0 = 421
    y0 = 287

    im0 = l_image[0]
    im1 = l_image[1]

    dx = 15
    dy = 15

    adx = np.arange(-dx, dx + 1, 1)
    ady = np.arange(-dy, dy + 1, 1)

    l_dx = []
    l_dy = []
    l_mse = []

    for dx in adx:
        for dy in ady:

            w2 = int(w/2)

            p0 = im0[x0 - w2:x0 + w2, y0 - w2: y0 + w2]
            p1 = im1[x0 + dx - w2:x0 + w2 + dx, y0 + dy - w2: y0 + w2+ dy]

            mse = mean_squared_error(p0.ravel(), p1.ravel())
            l_dx.append(dx)
            l_dy.append(dy)
            l_mse.append(mse)

    df_e = pd.DataFrame({'dx' : l_dx, 'dy': l_dy, 'mse' : l_mse})

    df_e = df_e.sort_values(by = 'mse')




    return df


vidcap = cv2.VideoCapture("C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\aagfhgtpmv.mp4")

df = detect_faces(vidcap)
vidcap.release()





####################################################################################
#
#   generate_frame_lines
#


def generate_frame_lines(vidcap):

    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = vidcap.get(cv2.CAP_PROP_FPS)

    nFrame = length
    iFrame = 0


    video = np.zeros((length, height, width, 3), dtype = np.uint8)


    for iFrame in range (nFrame):

        print(f"Processing {iFrame}...")

        success,image = vidcap.read()


        


        assert success, "Failed to read frame"

        (img_y, img_x, img_d) = image.shape

        video[iFrame] = image


    num_samples = 30000

    sample_length = 16
    sample_height = 1
    sample_width = 1

    data = np.zeros((num_samples, sample_length * sample_height * sample_width, 3))

    for i in range(num_samples):

        if i % 10000 == 0:
            print (i)
        sample_length_start = np.random.choice(length - sample_length)
        sample_start_height = np.random.choice(height - sample_height)
        sample_start_width = np.random.choice(width - sample_width)

        data_v = video[sample_length_start:sample_length_start + sample_length, sample_start_height:sample_start_height + sample_height, sample_start_width:sample_start_width + sample_width]
        data_v = data_v.reshape(-1, 3)

        data[i] = data_v

    data = data / 255

    return data



vidcap = cv2.VideoCapture("C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\aagfhgtpmv.mp4")

detect_faces(vidcap)

data0 = generate_frame_lines(vidcap)
vidcap.release()


vidcap = cv2.VideoCapture("C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\axfhbpkdlc.mp4")
data1 = generate_chunks(vidcap)

vidcap = cv2.VideoCapture("C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\bdgipnyobr.mp4")
data2 = generate_chunks(vidcap)

data = np.vstack([data0, data1, data2])








# https://www.tensorflow.org/tutorials/quickstart/advanced

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


from tensorflow.keras.initializers import Constant



from datetime import datetime


logdir = "C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(logdir)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')

    self.d2 = Dense(10, activation='softmax', kernel_initializer = Constant(value = 0.7))

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)




with writer.as_default():
    tf.summary.scalar   ('train_acc', train_accuracy.result()*100,      step = 0)
    tf.summary.histogram('model.d2' , model.d2.get_weights()[1],  step= 0

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  with writer.as_default():
    tf.summary.scalar   ('train_acc', train_accuracy.result()*100,      step = epoch +1)
    tf.summary.histogram('model.d2' , model.d2.get_weights()[1],  step=epoch +1)



  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'

  print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))

