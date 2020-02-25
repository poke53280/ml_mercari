



import numpy as np


from mp4_frames import get_test_dir
from face_detector import MTCNNDetector

from face_detector import _draw_box_integer_coords
from face_detector import _draw_single_point
from face_detector import _get_integer_coords_single_feature

from face_detector import _draw_face_bb

from mp4_frames import read_video
from featureline import find_two_consistent_faces

from featurerect import get_feature_sets
from featurerect import get_feature_lines

from featurerect import get_face_line
from line_sampler import get_line

from mp4_frames import get_part_dir
from mp4_frames import get_output_dir
from mp4_frames import read_metadata

import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error


######################################################################
#
#   draw_grid
#

def draw_grid(image_in, raster_w, raster_h, out):
    image = image_in.copy()
    for x in range(raster_w):
        for y in range(raster_h):
            p = out[x, y]
            _draw_box_integer_coords(image, p[0], p[1], 1)

    return image


######################################################################
#
#   get_sample_grid
#

def get_sample_grid (face, x_max, y_max, l_featureline, rW, raster_w, raster_h):

    assert raster_h > 0

    
    pc, vector = get_face_line(l_featureline, face, x_max, y_max)
    vector_p = np.array([vector[1], -vector[0]])

    if rW > 0:
        # Normalize to face feature size with slack
        vector = rW * vector
    else:
        # Normalize to raster size for dense sampling
        vector = (raster_w / np.sqrt(vector.dot(vector))) * vector
    
    # In either case, maintain aspect ratio
    rAspect = raster_h / raster_w
    vector_p = (rAspect / np.sqrt(vector_p.dot(vector_p))) * vector_p

    pA = pc - 0.5 * vector - 0.5 * vector_p

    # Rasterize

    step_x_vector = vector / (raster_w -1)

    # Avoid unnecessary zero division in case of degenerate axis.
    step_y_vector = 0 if raster_h == 1 else vector_p / (raster_h -1)

    anOut = np.zeros((raster_w, raster_h, 2), dtype = np.int32)


    for x in range(raster_w):
        for y in range(raster_h):
            p = pA + x * step_x_vector + y * step_y_vector
            anOut[x, y] = p.astype(np.int32)

    return anOut



######################################################################
#
#   sample
#

def sample(x_max, y_max, z_max, faces, l_featureline, rW, raster_w, raster_h):
    

    l_feature_0 = np.array((*_get_integer_coords_single_feature(x_max, y_max, faces[0], l_featureline[0]), 0))
    r_feature_0 = np.array((*_get_integer_coords_single_feature(x_max, y_max, faces[0], l_featureline[1]), 0))



    anBegin = get_sample_grid(faces[0], x_max, y_max, l_featureline, rW, raster_w, raster_h)
    anEnd   = get_sample_grid(faces[1], x_max, y_max, l_featureline, rW, raster_w, raster_h)

    aiGrid = np.zeros((raster_w, raster_h, z_max, 3), dtype = np.int32)

    for x in range(raster_w):
        for y in range(raster_h):
            p0 = anBegin[x, y]
            p1 = anEnd[x, y]

            p0 = (*p0, 0)
            p1 = (*p1, z_max - 1)

            l = get_line(np.array(p0), np.array(p1))

            assert l.shape[1] >= z_max

            l = l.astype(np.int32)
            
            l = np.swapaxes(l, 0, 1)

            num_points = l.shape[0]

            ai = np.linspace(0, num_points -1, num=z_max, endpoint=True) 

            an = ai.astype(np.int32)
                        
            l = l[an]

            aiGrid[x, y] = l

    l_x = aiGrid[:, :, :, 0].reshape(-1)
    l_y = aiGrid[:, :, :, 1].reshape(-1)
    l_z = aiGrid[:, :, :, 2].reshape(-1)

    # Handle out of bounds
    m = (l_x >= 0) & (l_x < x_max) & (l_y >= 0) & (l_y < y_max) & (l_z >= 0) & (l_z < z_max)

    l_x[~m] = l_y[~m] = l_z[~m] = 0

    return (anBegin, anEnd, l_x, l_y, l_z)


######################################################################
#
#   sample_feature
#

def sample_feature(video, faces, featureset, W, H, isDraw):

    invalid = (faces[0] is None) or (faces[1] is None)

    assert not invalid


    x_max = video.shape[2]
    y_max = video.shape[1]
    z_max = video.shape[0]

    video_size = z_max

   

    (anBegin, anEnd, l_x, l_y, l_z) = sample(x_max, y_max, z_max, faces, featureset, 0.0, W, H)

    anSample = video[l_z, l_y, l_x]

    anSample = anSample.reshape(W, H, video_size, 3)

    anSample = np.squeeze(anSample)

    anSample = np.swapaxes(anSample, 0, 1)

    if isDraw:

        image_begin = draw_grid(video[0], W, H, anBegin)
        image_end = draw_grid(video[z_max -1], W, H, anEnd)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.imshow(image_begin)
        ax2.imshow(image_end)
        ax3.imshow(anSample)
        plt.show()

    

    return anSample



######################################################################
#
#   sample_video
#

def sample_video(mtcnn_detector, video_path, isDraw):
    
    video_size = 32

    W = 256
    H = 1


    print (f"{video_path}")

    video = read_video(video_path, video_size)

    if video is None:
        return None

    if video.shape[0] != video_size:
        return None

    x_max = video.shape[2]
    y_max = video.shape[1]
    z_max = video.shape[0]

    faces = find_two_consistent_faces(mtcnn_detector, video)

    invalid = (faces[0] is None) or (faces[1] is None)

    if invalid:
        return None


    l_featuresets = [ ['l_mouth', 'r_mouth'], ['l_eye', 'r_eye'], ['bb_min', 'bb_max'], ['c_nose', 'r_eye'], ['l_mouth', 'c_nose'], ['f_min', 'f_max'],  ['l_eye', 'r_mouth']]
    
    l_sample = []

    for featureset in l_featuresets:
        anSample = sample_feature(video, faces, featureset, W, H, isDraw)

        l_feature0 = np.array((*_get_integer_coords_single_feature(x_max, y_max, faces[0], featureset[0]), 0))
        r_feature0 = np.array((*_get_integer_coords_single_feature(x_max, y_max, faces[0], featureset[1]), 0))

        vector = r_feature0 - l_feature0

        length_vector = np.sqrt(vector.dot(vector))
   
        anSample = straighten_sample(anSample, length_vector)

        anSample = anSample.reshape(-1)
        l_sample.append(anSample)

    assert len (l_sample) == len (l_featuresets)

    anData = np.stack(l_sample)

    return anData



######################################################################
#
#   sample_video_safe
#

def sample_video_safe(mtcnn_detector, video_path, isDraw):

    video_size = 32

    W = 256
    H = 1

    try:
        data = sample_video(mtcnn_detector, video_path, isDraw)
    except Exception as err:
        print(err)
        data = None

    return data


####################################################################################
#
#   process_part
#

def process_part(iPart):


    l_d = read_metadata(iPart)

    input_dir = get_part_dir(iPart)

    output_dir = get_output_dir()

    mtcnn_detector = MTCNNDetector()

    for o_set in l_d:

        l_samples = []

        original_path = input_dir / o_set[0]

        #print(f"{iPart}: {original_path.stem}...")

        r_data = sample_video_safe(mtcnn_detector, original_path, False)

        if r_data is None:
            print(f"{original_path.stem}: Bad original. Skipping set.")
            continue

        l_samples.append(r_data)

        for fake_path in o_set[1]:
            f_data = sample_video_safe(mtcnn_detector, input_dir / fake_path, False)

            if f_data is None:
                continue

            l_samples.append(f_data)

        if len (l_samples) >= 2:
            data = np.concatenate(l_samples)
            filename = f"p_{iPart}_{original_path.stem}.npy"
            output_path = output_dir / filename
            np.save(output_path, data)
        else:
            print(f"{original_path.stem}: No good fakes. Skipping set.")




####################################################################################
#
#   straighten_sample
#

def straighten_sample(anSample, length_vector):

    num_elements = anSample.shape[1]
    num_lines = anSample.shape[0]

    # center data
    cx = num_elements // 2

    Lh = int(length_vector / 2)


    if Lh > 0.4 * num_elements:
        Lh = int(0.4 * num_elements)

    if cx - Lh < 0:
        cx = Lh

    if cx + Lh > num_elements:
        cx = num_elements - Lh


    l_out = []

    l_out.append(anSample[0])


    for iline in range(num_lines -1):

        line0 = l_out[iline]
        line1 = anSample[iline + 1]

        line0_cut = line0[cx - Lh: cx + Lh]


        mse_low = 100000
        best_offset = 0

        for offset in range(-5, 6):
            line1_cut = line1[offset + cx - Lh: offset + cx + Lh]

            min_size = np.min([line1_cut.shape[0], line0_cut.shape[0]])

            if min_size == 0:
                print("Warning: Empty clip line")
                best_offset = 0
                break

            mse = mean_squared_error(line0_cut[:min_size], line1_cut[:min_size])

            if mse < mse_low:
                mse_low = mse
                best_offset = offset


        dst_idx = np.array(range(0, num_elements))

        src_idx = dst_idx + best_offset

        m_data = (src_idx >= 0) & (src_idx < num_elements)

        src_idx = src_idx[m_data]
        dst_idx = dst_idx[m_data]

        line1_out = np.zeros(shape = line1.shape, dtype = np.uint8)

        line1_out[dst_idx] = line1[src_idx]

        l_out.append(line1_out)

        # print(f"{iline}: {best_offset}: mse = {mse_low}")


    anSampleOut = np.stack(l_out)
    return anSampleOut




####################################################################################
#
#   run_one
#

def run_one():
    input_dir = get_part_dir(0)
    mtcnn_detector = MTCNNDetector()

    l_files = list (sorted(input_dir.iterdir()))

    l_files = [x for x in l_files if x.suffix == '.mp4']
    
    
    video_path = input_dir / "nrdnytturz.mp4"

    assert video_path.is_file()

    #video_path = l_files[126]

    video_size = 32

    W = 256
    H = 1

    video = read_video(video_path, video_size)

    x_max = video.shape[2]
    y_max = video.shape[1]
    z_max = video.shape[0]


    faces = find_two_consistent_faces(mtcnn_detector, video)

    featureset = ['l_mouth', 'r_mouth']

   
    anSample = sample_feature(video, faces, featureset, W, H, True)


    l_feature0 = np.array((*_get_integer_coords_single_feature(x_max, y_max, faces[0], featureset[0]), 0))
    r_feature0 = np.array((*_get_integer_coords_single_feature(x_max, y_max, faces[0], featureset[1]), 0))

    vector = r_feature0 - l_feature0

    length_vector = np.sqrt(vector.dot(vector))
   
    anSampleOut = straighten_sample(anSample, length_vector)

    anSample = anSample.reshape(-1)






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

    l_tasks = [2] # list (range(1))

    num_threads = 1

    # print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(process_part, l_tasks)







