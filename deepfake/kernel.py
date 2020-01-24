


import os

# os.system('pip install -q /kaggle/input/mtcnnpackage/mtcnn-0.1.0-py3-none-any.whl')
!pip install /kaggle/input/mtcnnpackage/mtcnn-0.1.0-py3-none-any.whl

       
import cv2
import numpy as np
import pandas as pd

from mtcnn.mtcnn import MTCNN
from sklearn.metrics import mean_squared_error
import pathlib
import random
import json
import time        
        


####################################################################################
#
#   get_sample_point
#

def get_sample_point(l_feature, width, height, delta):
    
    p = random.choice(l_feature)
    
    x = p[0] + np.random.choice(2 * delta) - delta
    y = p[1] + np.random.choice(2 * delta) - delta

    x = np.max([x, 0])
    x = np.min([x, width - 1])

    y = np.max([y, 0])
    y = np.min([y, height - 1])

    return (x, y)

####################################################################################
#
#   get_sample_point_no_feature
#

def get_sample_point_no_feature(width, height):
    x = np.random.choice(width)
    y = np.random.choice(height)
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
#   sample_line
#
#

def sample_line(length, height, width, sample_length, anFeatures):
    
    # Todo: More skewed, in-frame sampling. Keep angles.
    
    sample_length_start = np.random.choice(length - sample_length)

    sample_length_end = sample_length_start + (sample_length - 1)   # inclusive

    l_feature_start = anFeatures[sample_length_start]
    l_feature_end   = anFeatures[sample_length_end]

    isStart = len (l_feature_start) > 0
    isEnd    = len (l_feature_end) > 0

    if isStart and not isEnd:
        l_feature_end = l_feature_start
        isEnd = True

    if isEnd and not isStart:
        l_feature_start = l_feature_end
        isStart = True

    if isStart and isEnd:
        p0_2d = get_sample_point(l_feature_start, width, height, 3)
        p1_2d = get_sample_point(l_feature_end, width, height, 3)

    else:
        p0_2d = get_sample_point_no_feature(width, height)
        p1_2d = get_sample_point_no_feature(width, height)


    p0 = np.array([p0_2d[0], p0_2d[1], sample_length_start])
    p1 = np.array([p1_2d[0], p1_2d[1], sample_length_end])

    l = get_line(p0, p1)

    return l


####################################################################################
#
#   sample_video
#
#

def sample_video(video_real, video_fake, anFeatures, num_samples):

    isFake = video_fake is not None

    if isFake:
        assert video_real.shape == video_fake.shape

    length = video_real.shape[0]
    height = video_real.shape[1]
    width = video_real.shape[2]

    sample_length = 16
    sample_height = 1
    sample_width = 1

    data_real = np.zeros((num_samples, sample_length * sample_height * sample_width, 3), dtype = np.uint8)

    if isFake:
        data_fake = np.zeros((num_samples, sample_length * sample_height * sample_width, 3), dtype = np.uint8)

    
    iCollected = 0

    while iCollected < num_samples:

        if iCollected % 10000 == 0:
            print (f"{iCollected}/ {num_samples}")

        l = sample_line(length, height, width, sample_length, anFeatures)

        assert l.shape[1] >= 16

        l = np.swapaxes(l, 0, 1)

        l = l[:16]
        l = l.astype(np.int32)

        l_x = l[:, 0]
        l_y = l[:, 1]
        l_z = l[:, 2]

        sample_real = video_real[l_z, l_y, l_x]

        if isFake:
            sample_fake = video_fake[l_z, l_y, l_x]

            m = sample_real == sample_fake
            m = m.reshape(-1)

            nAll = m.shape[0]
            nFake = nAll - m.sum()

            if nFake == 0:
                continue
        
            data_fake[iCollected] = sample_fake
        
        data_real[iCollected] = sample_real

        iCollected = iCollected + 1


    if isFake:            
        return data_real, data_fake
    else:
        return data_real





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
#   descr
#

def descr(p, m):
    zM = m_desc(m)
    return f"{p}: {zM}"
"""c"""

####################################################################################
#
#   m_print
#

def m_print(p, m):
    print (descr(p, m))
"""c"""


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



###############################################################################
#
#    read_test_files
#
#

def read_test_files(input_dir):

    l_files = [x for x in input_dir.iterdir() if x.suffix == ".mp4"]

    l_data = []

    for idx_key, x in enumerate(l_files):

        print (x)
        vidcap = cv2.VideoCapture(str(x))
        video, anFeatures = read_image_and_features(vidcap)
        vidcap.release()


        data = sample_video(video, None, anFeatures, 100000)

        assert data.shape[0] == 100000
       
        anVidLo = np.empty(100000, dtype = np.uint8)
        anVidLo[:] = (idx_key % 256)
        
        anVidHi = np.empty(100000, dtype = np.uint8)
        anVidHi[:] = (idx_key // 256)

        data = data.reshape(data.shape[0], -1)

        data_line = np.hstack([anVidLo.reshape(-1, 1), anVidHi.reshape(-1, 1), data])

        l_data.append(data)

    return np.vstack(l_data)



input_dir = pathlib.Path("/kaggle/input/deepfake-detection-challenge/test_videos")
assert input_dir.is_dir()

data = read_test_files(input_dir)

np.save("samples.npy")


print("done.")



