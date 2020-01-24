

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
import time
import string



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

def sample_full_chunk(iPart, l_d, num_data_threshold):

    dir = get_part_dir(iPart)

    l_data = []

    num_real_data = 0

    while num_real_data < num_data_threshold:

        idx_key = np.random.choice(len(l_d))

        current = l_d[idx_key]

        x_real = current[0]

        l_fakes = current[1]

        if len(l_fakes) == 0:
            continue

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

        assert data_real.shape[0] == data_fake.shape[0]

        num_data = data_real.shape[0]

        if num_data == 0:
            continue

        anPart = np.empty(num_data, dtype = np.uint8)
        anPart[:] = iPart

        anVidLo = np.empty(num_data, dtype = np.uint8)
        anVidLo[:] = (idx_key % 256)
        
        anVidHi = np.empty(num_data, dtype = np.uint8)
        anVidHi[:] = (idx_key // 256)

        data_real = data_real.reshape(data_real.shape[0], -1)
        data_fake = data_fake.reshape(data_fake.shape[0], -1)


        data = np.hstack([anPart.reshape(-1, 1), anVidLo.reshape(-1, 1), anVidHi.reshape(-1, 1), data_real, data_fake])
        # Back
        #data_back = data_real_flat.reshape(-1, 16, 3)

        l_data.append(data)

        num_real_data = num_real_data + data_real.shape[0]

        print(f"Data collection {num_real_data}/ {num_data_threshold}")

    """c"""

    return np.vstack(l_data)



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

def read_metadata(iPart):
    
    p = get_part_dir(iPart)

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


    
    d = {}

    for x in l_original_files:
        d[x] = []

    assert len (l_fake_files) == len (l_original_files)

    t = list (zip (l_original_files, l_fake_files))

    for pair in t:
        assert pair[0] in d
        d[pair[0]].append(pair[1])            
            

    l_keys = list(d.keys())
    l_keys.sort()

    l_d = []

    for x in l_keys:
        l_d.append((x, d[x]))

    return l_d


input_dir = pathlib.Path(f"C:\\Users\\T149900\\Downloads")
assert input_dir.is_dir(), f"input dir {input_dir} not existing"

output_dir = pathlib.Path(f"C:\\Users\\T149900\\vid_out")
assert output_dir.is_dir(), f"output dir {output_dir} not existing"

####################################################################################
#
#   get_part_dir
#

def get_part_dir(iPart):
    s = input_dir / f"dfdc_train_part_{iPart:02}\\dfdc_train_part_{iPart}"
    assert s.is_dir()

    return s


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def sample_main(num_runs):
  
    for x in range(num_runs):

        iPart = np.random.choice([21])

        print(f"Sampling from part {iPart}...")

        l_d = read_metadata(iPart)

        anData = sample_full_chunk(iPart, l_d, 900000)

        timestr = time.strftime("%m%d_%H%M%S")

        zA = random.choice(string.ascii_lowercase)
        zB = random.choice(string.ascii_lowercase)
        zC = random.choice(string.ascii_lowercase)

        out = f"data_p{iPart}_{zA}{zB}{zC}_{timestr}.npy"

        print(f"Saving {out} ...")

        np.save(output_dir / out, anData)

    



