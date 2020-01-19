

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import signal

import matplotlib.animation as animation

from mtcnn.mtcnn import MTCNN
from sklearn.metrics import mean_squared_error
import pathlib
import torch
import random


def get_sample_point(l_feature, delta):
    p = random.choice(l_feature)
    x = p[0] + np.random.choice(2 * delta) - delta
    y = p[1] + np.random.choice(2 * delta) - delta
    return (x, y)



####################################################################################
#
#   sample_new
#
#
#   Todo: make use of features for more relevant sampling

def sample_new(video, anFeatures):
    num_samples = 30000

    length = video.shape[0]
    height = video.shape[1]
    width = video.shape[2]

    sample_length = 16
    sample_height = 1
    sample_width = 1

    data = np.zeros((num_samples, sample_length * sample_height * sample_width, 3))

    sample_length_start = np.random.choice(length - sample_length)

    sample_length_end = sample_length_start + 16

    l_feature_start = anFeatures[sample_length_start]
    l_feature_end   = anFeatures[sample_length_end]

    p0 = get_sample_point(l_feature_start, 3)
    p1 = get_sample_point(l_feature_end, 3)

    # Get 3d line

    # See sampler.py





####################################################################################
#
#   sample_basic
#


def sample_basic(video, anFeatures):

    num_samples = 30000

    length = video.shape[0]
    height = video.shape[1]
    width = video.shape[2]

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



####################################################################################
#
#   attic
#
#   Non-running code to fuzzy sample lines

def attic():

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
#   read_image_and_features
#

def read_image_and_features(vidcap):
    
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = vidcap.get(cv2.CAP_PROP_FPS)

    nFrame = length
    iFrame = 0

    detector = MTCNN()

    video = np.zeros((length, height, width, 3), dtype = np.uint8)

    l_p_image = []

    for iFrame in range (nFrame):

        print(f"Processing {iFrame}...")

        success,image = vidcap.read()

        video[iFrame] = image

        faces = detector.detect_faces(image)

        l_p = []

        for f in faces:
            if f['confidence'] < 0.8:
                continue

            l_p.append(f['keypoints']['left_eye'])
            l_p.append(f['keypoints']['right_eye'])
            l_p.append(f['keypoints']['nose'])
            l_p.append(f['keypoints']['mouth_left'])
            l_p.append(f['keypoints']['mouth_right'])

        l_p_image.append(l_p)

    return (video, l_p_image)





pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


p = pathlib.Path(r'C:\Users\T149900\Downloads\deepfake-detection-challenge\train_sample_videos')

assert p.is_dir()

file_list = []

for x in p.iterdir():
    if x.is_file() and x.suffix == '.mp4':
        file_list.append(x)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


vidcap = cv2.VideoCapture(str(file_list[13]))

video, anFeatures = read_image_and_features(vidcap)

vidcap.release()

data0 = generate_frame_lines(video, anFeatures)


vidcap = cv2.VideoCapture(str(file_list[19]))

video, anFeatures = read_image_and_features(vidcap)

vidcap.release()

data1 = generate_frame_lines(video, anFeatures)


sequence = np.vstack([data0, data1])

np.random.shuffle(sequence)

num_train = 50000
num_test = sequence.shape[0] - num_train


test_sequence = sequence[num_train:num_train + num_test]
test_sequence = test_sequence.reshape((test_sequence.shape[0], test_sequence.shape[1], 3))


sequence = sequence[:num_train]

num_samples = sequence.shape[0]
num_timesteps = sequence.shape[1]


# reshape input into [samples, timesteps, features]
sequence = sequence.reshape((num_samples, num_timesteps, 3))


# define model
model = Sequential()


model.add(LSTM(1024, activation='relu', input_shape=(num_timesteps, 3)))
model.add(RepeatVector(num_timesteps))
model.add(LSTM(1024, activation='relu', return_sequences=True))

#model.add(Dense(512))
#model.add(Dense(128))

model.add(TimeDistributed(Dense(3)))
model.compile(optimizer='adam', loss='mse')


# fit model
model.fit(sequence, sequence, epochs=2, verbose=1)




y_test = model.predict(test_sequence)

from sklearn.metrics import mean_squared_error

y_test = y_test.reshape(-1)
test_sequence = test_sequence.reshape(-1)

data_mse = mean_squared_error(y_test, test_sequence)

y_random = np.random.uniform(size = test_sequence.shape)

y_random = y_random.reshape((num_test, 16, 3))

y_random_predict = model.predict(y_random)

y_random_predict = y_random_predict.reshape(-1)
y_random = y_random.reshape(-1)

ran_mse = mean_squared_error(y_random, y_random_predict)

data_mse = data_mse * 1000
ran_mse =ran_mse * 1000

data_mse
ran_mse
