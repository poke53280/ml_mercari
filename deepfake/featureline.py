

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


####################################################################################
#
#   get_video_lines
#

def get_video_lines(x_max, y_max, z_max, face0, face1, v_max, num_samples):

    l_zFeature = [x for x in list(face0.keys()) if not "confidence" in x]

    d_lines = {}

    for zFeature in l_zFeature:

        start = (*_get_integer_coords_single_feature(x_max, y_max, face0, zFeature), 0)
        end = (*_get_integer_coords_single_feature(x_max, y_max, face1, zFeature), 31)

        start_x = start[0] 
        start_y = start[1] 

        start_z = 0
        end_z = 16

        end_x = end[0] 
        end_y = end[1]


        start_x = start_x + np.random.randint(-v_max, v_max + 1, size=num_samples)
        start_y = start_y + np.random.randint(-v_max, v_max + 1, size=num_samples)
        end_x = end_x + np.random.randint(-v_max, v_max + 1, size=num_samples)
        end_y = end_y + np.random.randint(-v_max, v_max + 1, size=num_samples)

        start_z = start_z + np.random.randint(0, 32 - 16, size=num_samples)
        end_z = start_z + 16

        
        assert (end_z < 32).all()

        an_x = np.array([start_x, end_x]).T
        an_x[an_x < 0] = 0
        an_x[an_x >= x_max] = x_max - 1

        an_y = np.array([start_y, end_y]).T
        an_y[an_y < 0] = 0
        an_y[an_y >= y_max] = y_max - 1

        an_z = np.array([start_z, end_z]).T
        assert (an_z[:, 1] - an_z[:, 0] == 16).all()

        an = np.hstack([an_x, an_y, an_z])

        anLines = rasterize_lines(an)

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

    v_max = 5
    max_permutations = v_max * v_max * v_max * v_max * (32 - 16)
    num_samples = int (0.4 * max_permutations)

    lines = get_video_lines(x_max, y_max, z_max, face0, face1, v_max, num_samples)

    l_data = []

    d_f = get_feature_converter()

    for zFeature in list(lines.keys()):
        samples = sample_cube(video, lines[zFeature]).reshape(-1, 16 * 3)

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


    video_real = read_video(video_real_path, 32)
    video_fake = read_video(video_fake_path, 32)

    (face0, face1) = find_two_consistent_faces(video_real)

    invalid = (face0 is None) or (face1 is None)

    if invalid:
        return None

    z_max = video_real.shape[0]
    x_max = video_real.shape[2]
    y_max = video_real.shape[1]

    v_max = 5
    max_permutations = v_max * v_max * v_max * v_max * (32 - 16)
    num_samples = int (0.4 * max_permutations)

    lines = get_video_lines(x_max, y_max, z_max, face0, face1, v_max, num_samples)

    l_data = []

    d_f = get_feature_converter()
    

    for zFeature in list(lines.keys()):
   
        real_samples = sample_cube(video_real, lines[zFeature]).reshape(-1, 16 * 3)
        fake_samples = sample_cube(video_fake, lines[zFeature]).reshape(-1, 16 * 3)

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

# Todo: Test for size diff and pos diff. Skip on failure.

def find_two_consistent_faces(video):

    m = MTCNNDetector()

    l_faces0 = m.detect(video[0])

    isSingleFace0 = len (l_faces0) == 1

    if not isSingleFace0:
        print("Not single face in frame 0. Skipping")
        return (None, None)

    l_faces1 = m.detect(video[31])

    isSingleFace1 = len (l_faces1) == 1

    if not isSingleFace1:
        print("Not single face in frame 31. Skipping")
        return (None, None)

    # Todo: Test for size diff and pos diff. Skip on failure.

    face0 = l_faces0[0]
    face1 = l_faces1[0]

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
                print(f"p_{iPart}_{str(original.stem)}_{str(fake.stem)}: No data.")
                pass
            else:
                file_out = output_dir / f"p_{iPart}_Train_{str(original.stem)}_{str(fake.stem)}.npy"
                np.save(file_out, data_train)


            data_test_real = sample_single(original)
            data_test_fake = sample_single(fake)

            isValid = (data_test_real is not None) and (data_test_fake is not None)

            if isValid:
                file_real_out = output_dir / f"p_{iPart}_Test_{str(original.stem)}_real.npy"
                np.save(file_real_out, data_test_real)

                file_fake_out = output_dir / f"p_{iPart}_Test_{str(fake.stem)}_fake.npy"
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

    l_tasks = list (range(25))

    num_threads = 60

    print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(process, l_tasks)


