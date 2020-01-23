

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

def sample_video(video_real, video_fake, anFeatures):

    assert video_real.shape == video_fake.shape

    num_samples = 100000

    length = video_real.shape[0]
    height = video_real.shape[1]
    width = video_real.shape[2]

    sample_length = 16
    sample_height = 1
    sample_width = 1

    data_real = np.zeros((num_samples, sample_length * sample_height * sample_width, 3), dtype = np.uint8)
    data_fake = np.zeros((num_samples, sample_length * sample_height * sample_width, 3), dtype = np.uint8)

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

        sample_fake = video_fake[l_z, l_y, l_x]

        m = sample_real == sample_fake
        m = m.reshape(-1)

        nAll = m.shape[0]
        nFake = nAll - m.sum()
        
        if nFake > 0:
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


    l_data = []

    num_real_data = 0

    while num_real_data < num_data_threshold:

        l_keys = list (d.keys())

        x_real = random.choice(l_keys)

        if len(d[x_real]) == 0:
            continue

        l_fakes = list (d[x_real])

        x_fake = random.choice(l_fakes)

        x_real = dir / x_real
        assert x_real.is_file()

        x_fake = dir / x_fake
        assert x_fake.is_file()
        
        vidcap = cv2.VideoCapture(str(x_real))

        video_real, anFeatures = read_image_and_features(vidcap)

        vidcap.release()

        vidcap = cv2.VideoCapture(str(x_fake))

        video_fake = read_image(vidcap)

        vidcap.release()

        if video_real.shape != video_fake.shape:
            continue

        data_real, data_fake = sample_video(video_real, video_fake, anFeatures)

        m = get_zero_rows(data_real)

        data_real = data_real[~m]
        data_fake = data_fake[~m]

        l_data_real.append(data_real)
        l_data_fake.append(data_fake)

        num_real_data = num_real_data + data_real.shape[0]

        print(f"Data collection {num_real_data}/ {num_data_threshold}")

    return np.vstack(l_data_real), np.vstack(l_data_fake)



####################################################################################
#
#   get_zero_rows
#

def get_zero_rows(data):
    nonzero = np.count_nonzero(data, axis = 1)

    m0 = nonzero[:, 0] == 0
    m1 = nonzero[:, 1] == 0
    m2 = nonzero[:, 2] == 0

    m = m0 & m1 & m2

    print(f"Discarding zero rows: {m_desc(m)}")

    return m


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


iPart = 0

dir = pathlib.Path(f"C:\\Users\\T149900\\Downloads\\dfdc_train_part_{iPart:02}\\dfdc_train_part_{iPart}")

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

for x in range(3):

    aiVideo, anData = sample_full_chunk(dir, d, 200000)

    out = f"data_{x:03}.npy"

    print(f"Saving {out} ...")

    np.save(dir / out, anDataFake)

    



