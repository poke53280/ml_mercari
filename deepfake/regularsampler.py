



from mp4_frames import get_test_dir
from face_detector import MTCNNDetector

from face_detector import _draw_box_integer_coords
from face_detector import _draw_single_point

from face_detector import _draw_face_bb

from mp4_frames import read_video
from featureline import find_two_consistent_faces

from featurerect import get_feature_sets
from featurerect import get_feature_lines

from featurerect import get_face_line
from line_sampler import get_line

import matplotlib.pyplot as plt
import cv2


######################################################################
#
#   draw_grid
#

def draw_grid(image_in, out):
    image = image_in.copy()
    for x in range(raster_w):
        for y in range(raster_h):
            p = out[x, y]
            _draw_box_integer_coords(image, p[0], p[1], 1)

    imgplot = plt.imshow(image)
    plt.show()


######################################################################
#
#   get_sample_grid
#

def get_sample_grid (face, rW, raster_w, raster_h):

    assert raster_h > 0

    rH = raster_h/ raster_w

    anFeatureSet = get_feature_sets()

    l_featureLine = anFeatureSet[0]

    x_max = video.shape[2]
    y_max = video.shape[1]

    
    p0, vector = get_face_line(l_featureLine, face, x_max, y_max, rW)

    length_line = np.sqrt(vector.dot(vector))

    p1 = p0 + vector

    vector_p = rH * np.array([vector[1], -vector[0]])

    vector_p.dot(vector)
    len_vector_p = np.sqrt(vector_p.dot(vector_p))

    pA = p0 - 0.5 * vector_p
    pB = pA + vector
    pC = pA + vector + vector_p
    pD = pA + vector_p
    
    #  D    C
    #
    #  A    B

    # Rasterize

    step_x_vector = vector / (raster_w -1)

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

def sample(video, mtcnn_detector, rW, raster_w, raster_h, isDraw):
    faces = find_two_consistent_faces(mtcnn_detector, video)

    anBegin = get_sample_grid(faces[0], rW, raster_w, raster_h)

    if isDraw:
        draw_grid(video[0], anBegin)

    anEnd   = get_sample_grid(faces[1], rW, raster_w, raster_h)

    if isDraw:
        draw_grid(video[31], anEnd)

    aiGrid = np.zeros((raster_w, raster_h, 32, 3), dtype = np.int32)

    for x in range(raster_w):
        for y in range(raster_h):
            p0 = anBegin[x, y]
            p1 = anEnd[x, y]
            l = get_line(np.array((*p0, 0)), np.array((*p1, 31)))

            assert l.shape[1] >= 32

            l = np.swapaxes(l, 0, 1)
            l = l[:32]
            l = l.astype(np.int32)

            aiGrid[x, y] = l


    l_x = aiGrid[:, :, :, 0].reshape(-1)
    l_y = aiGrid[:, :, :, 1].reshape(-1)
    l_z = aiGrid[:, :, :, 2].reshape(-1)

    # Handle out of bounds
    x_max = video.shape[0]
    y_max = video.shape[1]
    z_max = video.shape[2]


    m = (l_x >= 0) & (l_x < x_max) & (l_y >= 0) & (l_y < y_max) & (l_z >= 0) & (l_z < z_max)

    l_x[~m] = l_y[~m] = l_z[~m] = 0

    anSample = video[l_x, l_y, l_z]

    anSample = anSample.reshape(aiGrid.shape[0], aiGrid.shape[1], aiGrid.shape[2], 3)

    anSample = np.squeeze(anSample)

    return anSample



#####################################################


input_dir = get_test_dir()

mtcnn_detector = MTCNNDetector()

l_files = list (sorted(input_dir.iterdir()))

l_filenames = [str(x.name) for x in l_files]

video = read_video(l_files[109], 32)


rW = 3.0  # feature slack a

raster_w = 10
raster_h = 1


s = sample(video, mtcnn_detector, rW, raster_w, raster_h, True)




