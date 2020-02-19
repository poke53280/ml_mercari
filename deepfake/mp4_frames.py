
# Install on gcloud
# Deep learning, tf1.15, nvidia boot image


#sudo apt-get install liblzma-dev


# Python:
# https://tecadmin.net/install-python-3-7-on-ubuntu-linuxmint/


# sudo pip3.7 install --upgrade pip
# sudo pip3.7 install opencv-python
# sudo apt-get install -y libsm6 libxext6 libxrender-dev


# sudo pip3.7 install torch torchvision


# sudo pip3.7 install mtcnn

# sudo pip3.7 install pandas

# sudo pip3.7 install --upgrade setuptools
# sudo pip3.7 install tensorflow-gpu==1.15

# sudo pip3.7 install sklearn


# sudo apt-get install unzip

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

import cv2
import numpy as np
import pandas as pd

import pathlib
import random
import json
import time
import string
import os
import argparse
from multiprocessing import Pool
import datetime



####################################################################################
#
#   get_video_path_from_stem_and_ipart
#

def get_video_path_from_stem_and_ipart(stem, iPart):
    video_dir = get_part_dir(iPart)
    assert video_dir.is_dir()

    filename = video_dir / f"{stem}.mp4"
    assert filename.is_file(), f"Not a file: '{filename}'"

    return filename





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
#   read_video
#
#  
#  nFrames 0 - read all
# 

def read_video(filepath, nFrameMax):

    assert nFrameMax >= 0

    assert filepath.is_file()

    vidcap = cv2.VideoCapture(str(filepath))

    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = vidcap.get(cv2.CAP_PROP_FPS)

    if nFrameMax == 0:
        nFrame = length
    else:
        nFrame = np.min([nFrameMax, length])
    
    
    iFrame = 0

    video = np.zeros((nFrame, height, width, 3), dtype = np.uint8)

    for iFrame in range (nFrame):

        success,image = vidcap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        video[iFrame] = image

    vidcap.release()

    return video


######################################################################
#
#   get_test_video_filepaths
#

def get_test_video_filepaths():

    l_part_work = [0, 6, 7, 10, 18, 28, 37, 49]

    l_part_home = [2, 12, 20, 21, 30, 32, 35, 41, 49]

    l_part = l_part_work

    iPart = random.choice(l_part)

    video_dir = get_part_dir(iPart)
    l_d = read_metadata(iPart)

    idx = np.random.choice(len (l_d))

    original = l_d[idx][0]
    l_fakes = l_d[idx][1]

    assert len(l_fakes) > 0

    fake = random.choice(l_fakes)

    return (video_dir / original, video_dir / fake)


###################################################################################
#
#   detect_features
#

def detect_features(video):

    time0 = time.time()
    iFrame = 0
    nFrame = video.shape[0]

    detector = MTCNN()

    l_p_image = []

    for iFrame in range (nFrame):

        #if iFrame % 50 == 0:
        #print(f"Processing {iFrame}/{nFrame}")

        image = video[iFrame]  

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


    time1 = time.time()
    dtime = time1 - time0
    print(f"Detection processing time: {dtime}s")

    return l_p_image


####################################################################################
#
#   get_any_real_and_fake_video_from_part
#

def get_any_real_and_fake_video_from_part(iPart):
    l_d = read_metadata(iPart)
    dir = get_part_dir(iPart)

    num_videos = len(l_d)

    idx_key = np.random.choice(num_videos)
    
    current = l_d[idx_key]

    x_real = current[0]

    x_real = dir / x_real
    assert x_real.is_file()

    video_real = read_video(str(x_real))

    l_fakes = current[1]

    x_fake = random.choice(l_fakes)

    x_fake = dir / x_fake
    assert x_fake.is_file()

    video_fake = read_video(str(x_fake))

    return video_real, video_fake


####################################################################################
#
#   get_feature_from_part
#

def get_feature_from_part(iPart, if_detector):
    
    l_d = read_metadata(iPart)
    dir = get_part_dir(iPart)

    num_videos = len(l_d)
    #num_videos = 2

    print(f"Face detection on part {iPart}. {len(l_d)} original video(s). Processing {num_videos} videos in part")

    acc_features = []
   
    
    
    for idx_key in range(num_videos):

        current = l_d[idx_key]

        print(f"{current[0]}: {idx_key +1} of {len(l_d)}")

        x_real = current[0]

        x_real = dir / x_real
        assert x_real.is_file()

        video_real = read_video(str(x_real))

        num_frames = video_real.shape[0]
       
        start_processing = datetime.datetime.now()

        l_features = if_detector(video_real)

        end_processing = datetime.datetime.now()

        delta_processing = (end_processing - start_processing).total_seconds()

        assert len (l_features) == num_frames

        nSuccess = np.array(l_features).sum()

        rSuccess = 100.0 * nSuccess / num_frames

        print (f"Video time: {delta_processing}s. Success rate {rSuccess}%")
        
        acc_features.append(l_features)

    return acc_features



####################################################################################
#
#   if_detector_empty
#

def if_detector_empty(video_real):

    l_f = []

    for i in range(video_real.shape[0]):
        l_f.append(0)            

    return l_f


####################################################################################
#
#   sample_from_part
#

def sample_from_part(iPart, lines_per_video):

    l_d = read_metadata(iPart)

    print(f"Sampling from part {iPart}. {len(l_d)} original video(s). Sampling {lines_per_video} from each")

    dir = get_part_dir(iPart)

    l_data = []

    num_lines = 0

    for idx_key in range(len(l_d)):

        print(f"Processing original video {idx_key +1} of {len(l_d)}")

        current = l_d[idx_key]

        print(f"    original {current[0]}")

        x_real = current[0]

        l_fakes = current[1]

        if len(l_fakes) == 0:
            print("No fakes. Skipping.")
            continue

        x_fake = random.choice(l_fakes)

        print(f"    num fakes: {len(l_fakes)}. Using fake {x_fake}")

        x_real = dir / x_real
        assert x_real.is_file()

        x_fake = dir / x_fake
        assert x_fake.is_file()
        
        video_real = read_video(str(x_real))

        anFeatures = detect_features(video_real)

        video_fake = read_video(str(x_fake))

        if video_real.shape != video_fake.shape:
            continue

        data_real, data_fake = sample_video(video_real, video_fake, anFeatures, lines_per_video)

        m = get_zero_rows(data_real)

        assert (~m).all()

        assert data_real.shape[0] == data_fake.shape[0]

        num_data = data_real.shape[0]

        assert num_data == lines_per_video

        anPart = np.empty(num_data, dtype = np.uint8)
        anPart[:] = iPart

        anVidLo = np.empty(num_data, dtype = np.uint8)
        anVidLo[:] = (idx_key % 256)
        
        anVidHi = np.empty(num_data, dtype = np.uint8)
        anVidHi[:] = (idx_key // 256)

        data_real = data_real.reshape(data_real.shape[0], -1)
        data_fake = data_fake.reshape(data_fake.shape[0], -1)


        data = np.hstack([anPart.reshape(-1, 1), anVidLo.reshape(-1, 1), anVidHi.reshape(-1, 1), data_real, data_fake])

        num_lines = num_lines + data.shape[0]

        print(f"    done original {current[0]}. Lines collected: {data.shape[0]}. Collected in total: {num_lines}")

        l_data.append(data)
        

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
            file_exists = (p/x).is_file()
            if file_exists:
                l_real_files.append(x)
            else:
                print(f"Warning, missing original file {str(x)} in part {iPart}")

        if zLabel == "FAKE":

            original_file = txt_parsed[x]['original']

            file_exists = (p/x).is_file()
            orig_exists = (p/original_file).is_file()

            if file_exists and orig_exists:
                l_fake_files.append(x)
                l_original_files.append(original_file)
            else:
                print(f"Warning, missing original file {str(original_file)} and/or fake file {str(x)} in part {iPart}")

   
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




####################################################################################
#
#   get_code_dir
#

def get_code_dir():
    isLocal = os.name == 'nt'
    if isLocal:
        path = pathlib.Path("C:\\Users\\T149900\\Documents\\GitHub\\ml_mercari\\deepfake")
        
    else:
        path = pathlib.Path("/mnt/disks/tmp_mnt/data")
    
    assert path.is_dir(), f"output dir {output_dir} not existing"       

    return path



####################################################################################
#
#   _is_kaggle
#

def _is_kaggle():
    return pathlib.Path("/kaggle/input").is_dir()


####################################################################################
#
#   get_aux_dir
#

def get_aux_dir(zDir):
    isLocal = os.name == 'nt' 
    if isLocal:
        aux_dir = pathlib.Path(f"C:\\Users\\T149900")
    else:
        aux_dir = pathlib.Path("/mnt/disks/tmp_mnt/data")

    assert aux_dir.is_dir(), f"base aux dir {aux_dir} not existing"

    if len(zDir) == 0:
        return aux_dir

    aux_dir = aux_dir / zDir

    assert aux_dir.is_dir(), f"dir {aux_dir} not existing"

    return aux_dir


####################################################################################
#
#   get_output_dir
#

def get_output_dir():
    return get_aux_dir("vid_out")

####################################################################################
#
#   get_ready_data_dir
#

def get_ready_data_dir():
    return get_aux_dir("ready_data")

####################################################################################
#
#   get_pred0_dir
#

def get_pred0_dir():
    return get_aux_dir("pred0_out")


####################################################################################
#
#   get_chunk_dir()
#

def get_chunk_dir():
    return get_aux_dir("chunks")


####################################################################################
#
#   get_log_dir
#

def get_log_dir():
    return get_aux_dir("log_dir")


####################################################################################
#
#   get_model_dir
#

def get_model_dir():
    if _is_kaggle():
        p = pathlib.Path("/kaggle/input/models")
        assert p.is_dir()
        return p

    return get_aux_dir("mod_out")





####################################################################################
#
#   get_test_dir
#

def get_test_dir():
    if _is_kaggle():
        p = pathlib.Path("/kaggle/input/deepfake-detection-challenge/test_videos")
        assert p.is_dir()
        return p

    return get_aux_dir("test_dir")


####################################################################################
#
#   get_submission_dir
#

def get_submission_dir():
    if _is_kaggle():
        p = pathlib.Path("/kaggle/working")
        assert p.is_dir()
        return p


    return get_aux_dir("")


####################################################################################
#
#   get_part_dir
#

def get_part_dir(iPart):


    isLocal = os.name == 'nt'

    if isLocal:
        input_dir = pathlib.Path(f"C:\\Users\\T149900\\Downloads")
        s = input_dir / f"dfdc_train_part_{iPart:02}" / f"dfdc_train_part_{iPart}"
        
    else:
        input_dir = pathlib.Path("/mnt/disks/tmp_mnt/data")
        s = input_dir / f"dfdc_train_part_{iPart}"

    if s.is_dir():
        pass
    else:
        print(str(s))
        assert s.is_dir(), f"{s} not a directory"

    return s


####################################################################################
#
#   get_model_filepath
#

def get_model_filepath(zFeature, zModel_type, isAssertExists):
    
    modelfile = get_model_dir() / f"model_{zFeature}_{zModel_type}.h5"

    if isAssertExists:
        assert modelfile.is_file()
    
    return modelfile

####################################################################################
#
#   get_train_filepath
#

def get_train_filepath(zFeature, iPartMin, iPartMax, isAssertExists):
    data_file = get_ready_data_dir() / f"train_{zFeature}_p_{iPartMin}_p_{iPartMax}.npy"

    if isAssertExists:
        assert data_file.is_file() 
    
    return data_file


####################################################################################
#
#   get_train_filepath
#

def get_test_filepath(zFeature, iPartMin, iPartMax, isAssertExists):
    data_file = get_ready_data_dir() / f"test_{zFeature}_p_{iPartMin}_p_{iPartMax}.npy"

    
    if isAssertExists:
        assert data_file.is_file() 

    return data_file


####################################################################################
#
#   get_meta_test_filepath
#

def get_meta_test_filepath(iPartMin, iPartMax, isAssertExists):
    meta_file = get_ready_data_dir() / f"test_meta_p_{iPartMin}_p_{iPartMax}.pkl"

    if isAssertExists:
        assert meta_file.is_file() 
    return meta_file


####################################################################################
#
#   get_pred0_filepath
#

def get_pred0_filepath(zFeature, zModel_type, iPartMin, iPartMax, isAssertExists):
    output_file = get_pred0_dir() / f"pred_{zFeature}_p_{iPartMin}_p_{iPartMax}_{zModel_type}.npy"

    if isAssertExists:
        assert output_file.is_file() 
    return output_file


####################################################################################
#
#   process_part
#

def process_part(iPart):
    print(f"Sampling from part {iPart}...")

    lines_per_video = 100000

    anData = sample_from_part(iPart, lines_per_video)

    timestr = time.strftime("%m%d_%H%M%S")

    zA = random.choice(string.ascii_lowercase)
    zB = random.choice(string.ascii_lowercase)
    zC = random.choice(string.ascii_lowercase)

    out = f"data_p{iPart}_{zA}{zB}{zC}_{timestr}.npy"

    print(f"Saving {out} ...")

    output_dir = get_output_dir()

    np.save(output_dir / out, anData)

    print(f"Done part {iPart}")


####################################################################################
#
#   chunked_detect
#

def chunked_detect():
    iPart = 2
    l_d = read_metadata(iPart)

    current = l_d[0][0]

    x_real =  get_part_dir(iPart) / current

    assert x_real.is_file()

    print ("Reading video")

    video_real = read_video(str(x_real))

    num_frames = video_real.shape[0]

    chunk_size = 1

    num_chunks = num_frames//chunk_size + 1 * (num_frames % chunk_size != 0)

    needed_pad = num_chunks * chunk_size - num_frames

    if needed_pad > 0:

        video_last = video_real[-1]
        video_pad = np.stack([video_last for x in range(needed_pad)], axis=0)
        video_real = np.vstack([video_real, video_pad])

    aS = np.split(video_real, num_chunks)

    l_chunk = []

    for x in range(num_chunks):
        chunk = aS[x]
        print(chunk.shape)

        l_c = []

        for i in range (chunk_size):
            l_c.append(video_real[i])

        chunk_out = np.hstack(l_c)

        l_chunk.append(chunk_out)

    video_chunk = np.stack(l_chunk)

   
    detect_features(video_chunk)

    # CPU
    # chunk size 16 230 secs
    # chunk size 64  336
    # chunk size  1   231
    
    with Pool(8) as p:
        anFeatures = p.map(detect_features, [video_real[0:50], video_real[100:200], video_real[200:]])

    print (anFeatures)



def process_chunk(iPart):
    return get_feature_from_part(iPart, if_detector_haar)

####################################################################################
#
#   main
#

#if __name__ == '__main__':
   
    #with Pool(7) as p:
        #anFeatures = p.map(process_chunk, [0, 1, 2, 3, 4, 5, 6])


    # l_F = get_feature_from_part(iPart, if_detector_haar)

