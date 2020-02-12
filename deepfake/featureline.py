

from mp4_frames import read_metadata
from mp4_frames import get_part_dir
from mp4_frames import get_output_dir
from mp4_frames import read_video

from mp4_frames import get_test_video_filepaths

from line_sampler import rasterize_lines
from line_sampler import sample_cube

import matplotlib.pyplot as plt

from easy_face import create_diff_image
from face_detector import MTCNNDetector
from face_detector import _get_integer_coords_single_feature

import random

from multiprocessing import Pool

import numpy as np
import pandas as pd

import itertools


####################################################################################
#
#   get_video_lines
#
#  Samples returned in z_max length
#

def get_video_lines(x_max, y_max, z_max, face0, face1, v_max, num_samples):

    l_zFeature = [x for x in list(face0.keys()) if not "confidence" in x]

    d_lines = {}

    for zFeature in l_zFeature:

        start = (*_get_integer_coords_single_feature(x_max, y_max, face0, zFeature), 0)
        end = (*_get_integer_coords_single_feature(x_max, y_max, face1, zFeature), z_max - 1)

        start_x = start[0] 
        start_y = start[1] 

        start_z = np.zeros(num_samples).astype(np.int32)
        end_z = start_z + z_max - 1

        end_x = end[0] 
        end_y = end[1]


        start_x = start_x + np.random.randint(-v_max, v_max + 1, size=num_samples)
        start_y = start_y + np.random.randint(-v_max, v_max + 1, size=num_samples)
        end_x = end_x + np.random.randint(-v_max, v_max + 1, size=num_samples)
        end_y = end_y + np.random.randint(-v_max, v_max + 1, size=num_samples)


        an_x = np.array([start_x, end_x]).T
        an_x[an_x < 0] = 0
        an_x[an_x >= x_max] = x_max - 1

        an_y = np.array([start_y, end_y]).T
        an_y[an_y < 0] = 0
        an_y[an_y >= y_max] = y_max - 1

        an_z = np.array([start_z, end_z]).T

        an = np.hstack([an_x, an_y, an_z])

        anLines = rasterize_lines(an, z_max)

        d_lines[zFeature] = anLines
    return d_lines


####################################################################################
#
#   get_feature_converter
#

def get_feature_converter():
    d_f = {'bb_min': 0, 'bb_max' : 1, 'l_eye': 2, 'r_eye' : 3, 'c_nose': 4, 'l_mouth': 5, 'r_mouth': 6, 'f_min': 7, 'f_max': 8}
    return d_f


####################################################################################
#
#   sample_single
#

def sample_single(video_path):

    num_frames = 32
    v_max = 9

    video = read_video(video_path, 32)

    if video is None:
        return None

    if video.shape[0] == 0:
        return None

    (face0, face1) = find_two_consistent_faces(video)

    invalid = (face0 is None) or (face1 is None)

    if invalid:
        return None

    z_max = video.shape[0]
    x_max = video.shape[2]
    y_max = video.shape[1]

    max_permutations = v_max * v_max * v_max * v_max
    num_samples = int (0.4 * max_permutations)

    lines = get_video_lines(x_max, y_max, z_max, face0, face1, v_max, num_samples)

    l_data = []

    d_f = get_feature_converter()

    for zFeature in list(lines.keys()):
        samples = sample_cube(video, lines[zFeature]).reshape(-1, num_frames * 3)

        num = samples.shape[0]
        iF = d_f[zFeature]

        anF = np.array([iF] * num).reshape(num, 1).astype(np.uint8)

        combined_samples = np.hstack([anF, samples])
        l_data.append(combined_samples)

    anData = np.concatenate(l_data)
    return anData


####################################################################################
#
#   sample_pair
#

def sample_pair(video_real_path, video_fake_path):

    num_frames = 32
    v_max = 9

    video_real = read_video(video_real_path, num_frames)
    video_fake = read_video(video_fake_path, num_frames)

    (face0, face1) = find_two_consistent_faces(video_real)

    invalid = (face0 is None) or (face1 is None)

    if invalid:
        return None

    z_max = video_real.shape[0]
    x_max = video_real.shape[2]
    y_max = video_real.shape[1]

    
    max_permutations = v_max * v_max * v_max * v_max
    num_samples = int (0.4 * max_permutations)

    lines = get_video_lines(x_max, y_max, z_max, face0, face1, v_max, num_samples)

    l_data = []

    d_f = get_feature_converter()
    

    for zFeature in list(lines.keys()):
   
        real_samples = sample_cube(video_real, lines[zFeature]).reshape(-1, num_frames * 3)
        fake_samples = sample_cube(video_fake, lines[zFeature]).reshape(-1, num_frames * 3)

        num = real_samples.shape[0]
        iF = d_f[zFeature]

        anF = np.array([iF] * num).reshape(num, 1).astype(np.uint8)

        combined_samples = np.hstack([anF, real_samples, fake_samples])
        l_data.append(combined_samples)


    anData = np.concatenate(l_data)
    return anData



####################################################################################
#
#   find_two_consistent_faces
#

def find_two_consistent_faces(video):

    m = MTCNNDetector()

    l_faces0 = m.detect(video[0])
    l_faces1 = m.detect(video[31])


    l_bb_min = []
    l_bb_max = []

    l_confidence = []
    l_iFace = []

    l_idxFace = []

    for x in l_faces0 + l_faces1:
        l_bb_min.append(x['bb_min'])
        l_bb_max.append(x['bb_max'])
        l_confidence.append(x['confidence'])

    l_iFace.extend([0] * len (l_faces0))
    l_iFace.extend([1] * len (l_faces1))

    l_idxFace.extend(list (range(len(l_faces0))))
    l_idxFace.extend(list (range(len(l_faces1))))

    df_f = pd.DataFrame({'iFace' : l_iFace, 'iFaceidx': l_idxFace, 'confidence': l_confidence, 'bb_min' : l_bb_min, 'bb_max' : l_bb_max})
    
    x0 = df_f.bb_min.map(lambda x: x[0])
    y0 = df_f.bb_min.map(lambda x: x[1])

    x1 = df_f.bb_max.map(lambda x: x[0])
    y1 = df_f.bb_max.map(lambda x: x[1])

    L_x = x1 - x0
    L_y = y1 - y0

    c_x = 0.5 * (x1 + x0)
    c_y = 0.5 * (y1 + y0)

    A = L_x * L_y

    df_f = df_f.assign(c_x = c_x, c_y = c_y, A = A)

    df_f = df_f.drop(['bb_min', 'bb_max'], axis = 1)

    # Remove low confidence faces
    df_f = df_f[df_f.confidence > 0.9]


    df0 = df_f[df_f.iFace == 0].reset_index(drop = True)
    df1 = df_f[df_f.iFace == 1].reset_index(drop = True)

    if df0.shape[0] == 0 and df1.shape[0] == 0:
        print("No faces detected")
        return (None, None)

    if df0.shape[0] == 0:
        print("No face 0 detected")

        # Biggest face from (1) for both
        iLoc = df1.A.idxmax()
        iFace = int (df1.iloc[iLoc].iFaceidx)
        return (l_faces1[iFace], l_faces1[iFace])

    if df1.shape[0] == 0:
        print("No face 1 detected")

        # Biggest face from (1) for both
        iLoc = df0.A.idxmax()
        iFace = int (df0.iloc[iLoc].iFaceidx)
        return (l_faces0[iFace], l_faces0[iFace])


    assert df0.shape[0] > 0 and df1.shape[0] > 0

    # Pick largest face from 0 and match with best face on 1
    
    idx_large_face0 = df0.A.idxmax()

    face_info = df0.iloc[idx_large_face0]

    iFaceidx0 = int (face_info['iFaceidx'])

    face0 = l_faces0[iFaceidx0]

    d_x = np.abs(df1.c_x - face_info['c_x'])
    d_y = np.abs(df1.c_y - face_info['c_y'])
    d_A = np.abs(df1.A - face_info['A'])

    df1 = df1.assign(d_x = d_x, d_y = d_y, d_A = d_A)

    df1 = df1.assign(nA = np.searchsorted(np.unique(df1.d_A), df1.d_A))
    df1 = df1.assign(nX = np.searchsorted(np.unique(df1.d_x), df1.d_x))
    df1 = df1.assign(nY = np.searchsorted(np.unique(df1.d_y), df1.d_y))

    nScore = df1.nA + df1.nX + df1.nY

    df1 = df1.assign(nScore = nScore)

    idxmin = df1.nScore.idxmin()

    iFaceidx = int (df1.iloc[idxmin].iFaceidx)

    face1 = l_faces1[iFaceidx]

    return (face0, face1)



####################################################################################
#
#   process
#

def process(iPart):

    l_d = read_metadata(iPart)
    dir = get_part_dir(iPart)
    output_dir = get_output_dir()

    num_originals = len(l_d)

    for idx_key in range(num_originals):

        print(f"p_{iPart}: Processing original {idx_key + 1} / {num_originals}")

        current = l_d[idx_key]

        original =  dir / current[0]
        fake = dir / random.choice(current[1])
        
        if (original.is_file() and fake.is_file()):
            data_train = sample_pair(original, fake)

            if data_train is None:
                print(f"Line_Pair_p_{iPart}_{str(original.stem)}_{str(fake.stem)}: No data.")
                pass
            else:
                file_out = output_dir / f"Line_Pair_p_{iPart}_{str(original.stem)}_{str(fake.stem)}.npy"
                np.save(file_out, data_train)


            data_test_real = sample_single(original)
            data_test_fake = sample_single(fake)

            isValid = (data_test_real is not None) and (data_test_fake is not None)

            if isValid:
                file_real_out = output_dir / f"Line_Test_p_{iPart}_{str(original.stem)}_real.npy"
                np.save(file_real_out, data_test_real)

                file_fake_out = output_dir / f"Line_Test_p_{iPart}_{str(fake.stem)}_fake.npy"
                np.save(file_fake_out, data_test_fake)



####################################################################################
#
#   __main__
#

if __name__ == '__main__':
    outdir_test = get_output_dir()
    assert outdir_test.is_dir()

    file_test = outdir_test / "test_out_cubes.txt"
    nPing = file_test.write_text("ping")
    assert nPing == 4

    l_tasks = list (range(50))

    num_threads = 50

    print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(process, l_tasks)


    print(f"featureline all done.")

