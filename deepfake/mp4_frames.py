

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

####################################################################################
#
#   get_sample_point
#
#

def get_sample_point(l_feature, width, height, delta):
    p = random.choice(l_feature)
    x = p[0] + np.random.choice(2 * delta) - delta
    y = p[1] + np.random.choice(2 * delta) - delta
    x = np.max([x, 0])
    x = np.min([x, width - 1])

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

        sample_length_end = sample_length_start + (16 - 1)   # inclusive

        l_feature_start = anFeatures[sample_length_start]
        l_feature_end   = anFeatures[sample_length_end]

        if (len(l_feature_start) == 0) or (len(l_feature_end) == 0):
            continue

        p0_2d = get_sample_point(l_feature_start, width, height, 3)
        p1_2d = get_sample_point(l_feature_end, width, height, 3)

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

        # assumes at least one fake
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

def sample_full_chunk(dir, d, num_data_threshold):


    l_data_real = []
    l_data_fake = []

    num_real_data = 0

    while num_real_data < num_data_threshold:

        x = random.choice(list (d.keys()))

        print (x)
        l_fake = d[x]
        x = dir / x

        l_fake = [dir / x for x in l_fake]

        vidcap = cv2.VideoCapture(str(x))

        video, anFeatures = read_image_and_features(vidcap)

        vidcap.release()

        l_fake_video = []

        # For mem reasons
        if len(l_fake) > 3:
            l_fake_3 = random.sample(l_fake, k = 3)
        else:
            l_fake_3 = l_fake

        for fake in l_fake_3:

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

    return np.vstack(l_data_real), np.vstack(l_data_fake)



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


iPart = 21

dir = pathlib.Path(f"C:\\Users\\T149900\\Downloads\\dfdc_train_part_21\\dfdc_train_part_{iPart}")

assert dir.is_dir()


l_real_files, l_fake_files, l_original_files = read_metadata(dir)

d = {}

for x in l_original_files:
    d[x] = []

assert len (l_fake_files) == len (l_original_files)

t = list (zip (l_original_files, l_fake_files))

for pair in t:
    assert pair[0] in d
    d[pair[0]].append(pair[1])

# Test, train split

s = set (d.keys())

num_originals = len (s)

rSplit = 0.3

num_splitA = int (num_originals * rSplit + .5)
num_splitB = num_originals - num_splitA

assert num_originals == num_splitA + num_splitB

keys_splitA = set (random.sample(s, k = num_splitA))
keys_splitB = s - keys_splitA

d_A = {}
d_B = {}

for x in list (keys_splitA):
    d_A[x] = d[x]

for x in list (keys_splitB):
    d_B[x] = d[x]


for x in range(10):

    anDataReal, anDataFake = sample_full_chunk(dir, d_A, 1000000)

    np.save(dir / f"real_data_A_{x:03}.npy", anDataReal)
    np.save(dir / f"fake_data_A_{x:03}.npy", anDataFake)

    anDataReal, anDataFake = sample_full_chunk(p, d_B, 1000000)

    np.save(dir / f"real_data_B_{x:03}.npy", anDataReal)
    np.save(dir / f"fake_data_B_{x:03}.npy", anDataFake)

