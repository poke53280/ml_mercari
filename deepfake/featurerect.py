

from mp4_frames import read_metadata
from mp4_frames import get_part_dir
from mp4_frames import get_output_dir
from mp4_frames import read_video
from mp4_frames import get_test_video_filepaths

from line_sampler import get_line
from line_sampler import rasterize_lines
from line_sampler import sample_cube

import matplotlib.pyplot as plt

from easy_face import create_diff_image
from face_detector import MTCNNDetector
from face_detector import _get_integer_coords_single_feature

from featureline import find_two_consistent_faces

import random
from multiprocessing import Pool
import numpy as np





####################################################################################
#
#   get_feature_sets
#

def get_feature_sets():
    anFeatureSet = [['l_mouth', 'r_mouth'], ['l_mouth', 'r_eye'], ['bb_min', 'bb_max'], ['l_eye', 'r_eye'], ['c_nose', 'bb_max']]
    return anFeatureSet


####################################################################################
#
#   sample_image_single
#

def sample_image_single(iPart, video_path, isFake):

    W = 128

    video = read_video(video_path, 32)

    faces = find_two_consistent_faces(video)

    invalid = (faces[0] is None) or (faces[1] is None)

    if invalid:
        return


    x_max = video.shape[2]
    y_max = video.shape[1]

    anFeatureSet = get_feature_sets()

    for l_feature_set in anFeatureSet:
        l_image = []

        for i in range(100):

            anLines = get_feature_lines(x_max, y_max, faces, l_feature_set, W, 5)

            if anLines is None:
                continue
        
            anImage = sample_feature_image(anLines, video)
            l_image.append(anImage)

        if len (l_image) > 0:

            if isFake:
                zClass = 'fake'
            else:
                zClass = 'real'

            anImageSet = np.stack(l_image)
            zFilename = f"IMG_p_{iPart}_{video_path.name}_{l_feature_set[0]}_{l_feature_set[1]}_{zClass}"
            np.save(get_output_dir() / zFilename, anImageSet)
   



####################################################################################
#
#   sample_image_pair
#

def sample_image_pair(iPart, video_real_path, video_fake_path):

    W = 128

    video_real = read_video(video_real_path, 32)

    faces = find_two_consistent_faces(video_real)

    invalid = (faces[0] is None) or (faces[1] is None)

    if invalid:
        return


    video_fake = read_video(video_fake_path, 32)

    x_max = video_real.shape[2]
    y_max = video_real.shape[1]

    anFeatureSet = get_feature_sets()

    for l_feature_set in anFeatureSet:
        l_image_real = []
        l_image_fake = []

        for i in range(100):

            anLines = get_feature_lines(x_max, y_max, faces, l_feature_set, W, 5)

            if anLines is None:
                continue
        
            anImageReal = sample_feature_image(anLines, video_real)
            l_image_real.append(anImageReal)

            anImageFake = sample_feature_image(anLines, video_fake)
            l_image_fake.append(anImageFake)
            
        if (len (l_image_real) > 0) and (len (l_image_fake) > 0):

            anImageSetReal = np.stack(l_image_real)
            anImageSetFake = np.stack(l_image_fake)

            zFilenameReal = f"IMG_p_{iPart}_{video_real_path.name}_{video_fake_path.name}_{l_feature_set[0]}_{l_feature_set[1]}_real"
            zFilenameFake = f"IMG_p_{iPart}_{video_real_path.name}_{video_fake_path.name}_{l_feature_set[0]}_{l_feature_set[1]}_fake"

            np.save(get_output_dir() / zFilenameReal, anImageSetReal)
            np.save(get_output_dir() / zFilenameFake, anImageSetFake)


####################################################################################
#
#   sample_feature_image
#

def sample_feature_image(anLines, video):
    l_sample = []

    for l in anLines:

        l_x = l[:, 0]
        l_y = l[:, 1]
        l_z = l[:, 2]

        r_sample = video[l_z, l_y, l_x]
        l_sample.append(r_sample)

    anSamples = np.stack(l_sample)

    return anSamples



####################################################################################
#
#   get_feature_lines
#

def get_feature_lines(x_max, y_max, faces, l_featureLine, W, d_radius):

    face0 = faces[0]
    face1 = faces[1]

    l0 = get_face_line(l_featureLine, face0, x_max, y_max, W)
    l1 = get_face_line(l_featureLine, face1, x_max, y_max, W)

    if (l0[0] is None) or (l1[0] is None):
        return None

    l0 = (l0[0] + np.random.randint(-d_radius, d_radius + 1), l0[1] + np.random.randint(-d_radius, d_radius + 1))
    l1 = (l1[0] + np.random.randint(-d_radius, d_radius + 1), l1[1] + np.random.randint(-d_radius, d_radius + 1))

    p0       = l0[0]
    cVector0 = l0[1]

    p1       = l1[0]
    cVector1 = l1[1]

    t_step = 1/W

    l_l = []

    for x in range(W):
        t = x * t_step
        pos0 = p0 + t * cVector0
        pos1 = p1 + t * cVector1

        p_start = np.array([pos0[0], pos0[1], 0])
        p_end   = np.array([pos1[0], pos1[1], 32])

        l = get_line(p_start, p_end)
    
        assert l.shape[1] >= 32, f"Line is short: {l.shape[1]}"

        l = np.swapaxes(l, 0, 1)
        l = l[:32]
        l = l.astype(np.int32)

        l_l.append(l)


    anLines = np.stack(l_l)

    return anLines


####################################################################################
#
#   get_centered_line
#

def get_centered_line(p0_l, p0_r, x_max, y_max, W):
    p0_c = .5 * (p0_r + p0_l)

    lrVector = (p0_r - p0_l)
    cHalfVector = W/2.0 * (lrVector / np.sqrt(np.dot(lrVector, lrVector)))


    t_x_min = - p0_c[0]/ cHalfVector[0]
    t_x_max = (x_max - p0_c[0])/ cHalfVector[0]

    t_y_min = - p0_c[1]/ cHalfVector[1]
    t_y_max = (y_max - p0_c[1])/ cHalfVector[1]

    t_min = np.max([np.min([t_x_min, t_x_max]), np.min([t_y_min, t_y_max])])
    t_max = np.min([np.max([t_x_min, t_x_max]), np.max([t_y_min, t_y_max])])

    d_min = -1
    d_max = 1

    if t_min > -1:
        d_min = t_min
        d_max = d_min + 2

    if t_max < 1:
        d_max = t_max
        d_min = d_max - 2

    if d_max < 1 or d_min > -1:
        print("No line possible")
        return (None, None)

    p0_min = p0_c + d_min * cHalfVector
    p0_max = p0_c + d_max * cHalfVector

    cL = p0_max - p0_min
    
    L = np.sqrt(np.dot(cL, cL))
    L_lrVector = np.sqrt(np.dot(lrVector,lrVector))

    rRatio =  L * L_lrVector / np.dot(cL, lrVector)

    assert np.abs(rRatio > 0.99)
    assert L/W > 0.999 
    assert L/W < 1.001 

    return (p0_min, 2 * cHalfVector)


####################################################################################
#
#   get_face_line
#

def get_face_line(l_featureLine, face, x_max, y_max, W):

    p0_l = np.array(_get_integer_coords_single_feature(x_max, y_max, face, l_featureLine[0])).astype(np.float32)
    p0_r = np.array(_get_integer_coords_single_feature(x_max, y_max, face, l_featureLine[1])).astype(np.float32)
    
    return get_centered_line(p0_l, p0_r, x_max, y_max, W)


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
            sample_image_pair(iPart, original, fake)

            sample_image_single(iPart, original, False)
            sample_image_single(iPart, fake, True)



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



aturerect.py:236: RuntimeWarning: divide by zero encountered in float_scalars
  t_y_min = - p0_c[1]/ cHalfVector[1]
featurerect.py:237: RuntimeWarning: divide by zero encountered in double_scalars


