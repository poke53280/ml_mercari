



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
#   etcf
#

def etcf (mtcnn_detector, image, face, isDraw):

    if isDraw:
        mtcnn_detector.draw(image, [face])

    rW = 2.0  # Feature slack 

    raster_w = 10
    raster_h = 3

    assert raster_h > 1


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

    #
    #  D       C
    #
    #  A       B
    #

    # Rasterize

    step_x_vector = vector / (raster_w -1 )
    step_y_vector = vector_p / (raster_h -1 )

    anOut = np.zeros((raster_w, raster_h, 2), dtype = np.int32)


    for x in range(raster_w):
        for y in range(raster_h):
            p = pA + x * step_x_vector + y * step_y_vector
            anOut[x, y] = p.astype(np.int32)

    if isDraw:
        for x in range(raster_w):
            for y in range(raster_h):
                p = anOut[x, y]
                _draw_box_integer_coords(image, p[0], p[1], 1)

        imgplot = plt.imshow(image)
        plt.show()

    return anOut


input_dir = get_test_dir()

mtcnn_detector = MTCNNDetector()

l_files = list (sorted(input_dir.iterdir()))

l_filenames = [str(x.name) for x in l_files]

video = read_video(l_files[197], 32)

faces = find_two_consistent_faces(mtcnn_detector, video)


anBegin = etcf(mtcnn_detector, video[0].copy(), faces[0], True)
anEnd   = etcf(mtcnn_detector, video[31].copy(), faces[1], True)

# 0,0

p0 = anBegin[0, 0]
p1 = anEnd[0, 0]


get_line(np.array((*p0, 0)), np.array((*p1, 31))).shape


# ...


