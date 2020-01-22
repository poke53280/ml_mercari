

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
import json

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model




####################################################################################
#
#   get_sample_point
#
#

def get_sample_point(l_feature, delta):
    p = random.choice(l_feature)
    x = p[0] + np.random.choice(2 * delta) - delta
    y = p[1] + np.random.choice(2 * delta) - delta
    return (x, y)


####################################################################################
#
#   get_line
#
#

def get_line(p0, p1):

    dp = p1 - p0
    dp = np.abs(dp)

    num_steps = np.max(dp)

    # t element of [0, 1]

    step_size = 1 / num_steps

    ai = np.arange(start = 0, stop = 1 + step_size, step = step_size)

    ai_t = np.tile(ai, 3).reshape(-1, ai.shape[0])


    p = (p1 - p0).reshape(3, -1) * ai_t

    p = p + p0.reshape(3, -1)

    p = np.round(p)

    return p


####################################################################################
#
#   sample_video
#
#

def sample_video(video_real, l_video_fake, anFeatures):

    for video_fake in l_video_fake:
        assert video_real.shape == video_fake.shape

    num_samples = 100000

    length = video_real.shape[0]
    height = video_real.shape[1]
    width = video_real.shape[2]

    sample_length = 16
    sample_height = 1
    sample_width = 1

    data_real = np.zeros((num_samples, sample_length * sample_height * sample_width, 3))
    data_fake = np.zeros((num_samples, sample_length * sample_height * sample_width, 3))

    for i in range(num_samples):

        if i % 1000 == 0:
            print (i)

        sample_length_start = np.random.choice(length - sample_length)

        sample_length_end = sample_length_start + 16

        l_feature_start = anFeatures[sample_length_start]
        l_feature_end   = anFeatures[sample_length_end]

        if (len(l_feature_start) == 0) or (len(l_feature_end) == 0):
            continue

        p0_2d = get_sample_point(l_feature_start, 3)
        p1_2d = get_sample_point(l_feature_end, 3)

        p0 = np.array([p0_2d[0], p0_2d[1], sample_length_start])
        p1 = np.array([p1_2d[0], p1_2d[1], sample_length_end])

        l = get_line(p0, p1)

        assert l.shape[1] >= 16

        l = np.swapaxes(l, 0, 1)

        l = l[:16]
        l = l.astype(np.int32)

        l_x = l[:, 0]
        l_y = l[:, 1]
        l_z = l[:, 2]

        sample_real = video_real[l_z, l_y, l_x]

        video_fake = random.choice(l_video_fake)

        sample_fake = video_fake[l_z, l_y, l_x]

        m = sample_real == sample_fake
        m = m.reshape(-1)

        nAll = m.shape[0]
        nFake = nAll - m.sum()
        rFake = nFake / nAll
        
        if rFake > 0.3:
            data_fake[i] = sample_fake

        data_real[i] = sample_real

    return data_real, data_fake




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


###################################################################################
#
#   read_image
#

def read_image(vidcap):
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = vidcap.get(cv2.CAP_PROP_FPS)

    nFrame = length
    iFrame = 0

    video = np.zeros((length, height, width, 3), dtype = np.uint8)

    for iFrame in range (nFrame):

        success,image = vidcap.read()

        video[iFrame] = image

    return video


###################################################################################
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

        if iFrame % 50 == 0:
            print(f"Processing {iFrame}/{nFrame}")

        success,image = vidcap.read()

        video[iFrame] = image

        faces = detector.detect_faces(image)

        l_p = []

        for f in faces:
            if f['confidence'] < 0.5:
                continue

            l_p.append(f['keypoints']['left_eye'])
            l_p.append(f['keypoints']['right_eye'])
            l_p.append(f['keypoints']['nose'])
            l_p.append(f['keypoints']['mouth_left'])
            l_p.append(f['keypoints']['mouth_right'])

        l_p_image.append(l_p)

    return (video, l_p_image)


####################################################################################
#
#   sample_full_chunk
#

def sample_full_chunk(p, d, num_data_threshold):


    l_data_real = []
    l_data_fake = []

    num_real_data = 0

    while num_real_data < num_data_threshold:

        x = random.choice(list (d.keys()))

        print (x)
        l_fake = d[x]
        x = p / x

        l_fake = [p / x for x in l_fake]

        vidcap = cv2.VideoCapture(str(x))

        video, anFeatures = read_image_and_features(vidcap)

        vidcap.release()

        l_fake_video = []

        # For mem reasons
        for fake in l_fake[:3]:

            vidcap = cv2.VideoCapture(str(fake))

            video_fake = read_image(vidcap)

            vidcap.release()

            l_fake_video.append(video_fake)


        data_real, data_fake = sample_video(video, l_fake_video, anFeatures)

        data_real = clean_sample_data(data_real)
        data_fake = clean_sample_data(data_fake)

        l_data_real.append(data_real)
        l_data_fake.append(data_fake)

        num_real_data = num_real_data + data_real.shape[0]

        print(f"Data collection {num_real_data}/ {num_data_threshold}")

    return np.vstack(data_real), np.vstack(data_fake)



####################################################################################
#
#   clean_sample_data
#

def clean_sample_data(data):
    nonzero = np.count_nonzero(data, axis = 1)

    m0 = nonzero[:, 0] == 0
    m1 = nonzero[:, 1] == 0
    m2 = nonzero[:, 2] == 0

    m = m0 & m1 & m2

    print(m_desc(m))

    data = data[~m]
    data = data / 255.0
    return data


####################################################################################
#
#   read_metadata
#

def read_metadata(p):
    
    assert p.is_dir()

    metadata = p / "metadata.json" 

    assert metadata.is_file()

    txt = metadata.read_text()

    txt_parsed = json.loads(txt)

    l_files = list (txt_parsed.keys())

    l_real_files = []
    l_fake_files = []
    l_original_files = []


    for x in l_files:
        zLabel = txt_parsed[x]['label']
        if zLabel == "REAL":
            l_real_files.append(x)
        if zLabel == "FAKE":
            l_fake_files.append(x)
            l_original_files.append(txt_parsed[x]['original'])


    return l_real_files, l_fake_files, l_original_files



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


p = pathlib.Path(r'C:\Users\T149900\Downloads\dfdc_train_part_00\dfdc_train_part_0')

l_real_files, l_fake_files, l_original_files = read_metadata(p)

d = {}

for x in l_original_files:
    d[x] = []

assert len (l_fake_files) == len (l_original_files)

t = list (zip (l_original_files, l_fake_files))

for pair in t:
    assert pair[0] in d
    d[pair[0]].append(pair[1])


anDataReal, anDataFake = sample_full_chunk(p, d, 300000)


device = 'cpu'


sequence = anDataReal

outfile = p / "real_data.npy"
np.save(outfile, sequence)


np.random.shuffle(sequence)

num_train = int (0.7 * sequence.shape[0])
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

model.add(LSTM(2048, activation='relu', input_shape=(num_timesteps, 3)))
model.add(RepeatVector(num_timesteps))
model.add(LSTM(2048, activation='relu', return_sequences=True))

model.add(Dense(512))
model.add(Dense(128))

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
