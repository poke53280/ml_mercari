

import numpy as np
import pandas as pd
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from blazeface import BlazeFace

from mp4_frames import get_part_dir

from mp4_frames import read_metadata
from mp4_frames import read_video

from image_grid import GetGrid2DBB
from image_grid import GetSubVolume2D

import matplotlib.pyplot as plt

import cv2


######################################################################
#
#   get_test_frame
#

def get_test_frame():
    video_dir = get_part_dir(0)
    l_d = read_metadata(0)

    idx = np.random.choice(len (l_d))

    current = l_d[idx][0]

    filepath = video_dir / current

    video = read_video(filepath)

    image = video[110]
    return image




#################################################################################
#
#   main_get_art_arg
#

def main_get_art_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", "-p", help="parts to sample from", required = True)

    args = parser.parse_args()

    _= get_output_dir()

    iPart = int(args.part)

    return iPart

######################################################################
#
#   BlazeDetector
#

class BlazeDetector():
    def __init__(self, weight_path):
        assert weight_path.is_dir()
        gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = BlazeFace().to(gpu)
        self.net.load_weights(weight_path / "blazeface.pth")
        self.net.load_anchors(weight_path / "anchors.npy")

        self.net.min_score_thresh = 0.75
        self.net.min_suppression_threshold = 0.3

    def detect(self, img):
        assert img.shape == (128, 128, 3)
        detections = self.net.predict_on_image(img)

        l_face = []

        for anFace in detections:
            face  = {}

            face['bb_min']  = (anFace[1].item(), anFace[0].item())
            face['bb_max']  = (anFace[3].item(), anFace[2].item())

            face['r_eye']   = (anFace[4].item(), anFace[5].item())
            face['l_eye']   = (anFace[6].item(), anFace[7].item())

            face['nose']    = (anFace[8].item(), anFace[9].item())
            face['mouth']   = (anFace[10], anFace[11])

            face['r_ear']   = (anFace[12].item(), anFace[13].item())        
            face['l_ear']   = (anFace[14].item(), anFace[15].item())  

            face['confidencee'] = anFace[16].item()
            
            l_face.append(face)

        return l_face


def draw_face_bb(img_sub, face):

    x_shape = img_sub.shape[0]
    y_shape = img_sub.shape[1]

    x0 = int (face['bb_min'][0] * x_shape)
    y0 = int (face['bb_min'][1] * y_shape)

    x1 = int (face['bb_max'][0] * x_shape)
    y1 = int (face['bb_max'][1] * y_shape)

    cv2.rectangle(img_sub, (x0, y0), (x1, y1), (255,0,0), 2)



def _draw_single_feature(img_sub, face, zFeature, rect_size):
    
    assert rect_size > 0

    x_shape = img_sub.shape[0]
    y_shape = img_sub.shape[1]

    x = int (face[zFeature][0] * x_shape)
    y = int (face[zFeature][1] * y_shape)

    x0 = x - rect_size
    y0 = y - rect_size

    x1 = x + rect_size
    y1 = y + rect_size

    cv2.rectangle(img_sub, (x0, y0), (x1, y1), (255,0,0), 2)

       
   
       
weight_path = pathlib.Path(f"C:\\Users\\T149900\\Documents\\GitHub\\ml_mercari\\deepfake")
assert weight_path.is_dir()


b = BlazeDetector(weight_path)


image_base = get_test_frame()

image = image_base.copy()



#img = cv2.resize(video[1], (128, 128))
#i = b.detect(img)

sample_size = 500
rOverlap = 1.3

l_bb = GetGrid2DBB(image.shape[0], image.shape[1], sample_size, 1.3)

l_face = []

for bb in l_bb:
    
    img_sub = GetSubVolume2D(image, bb)
    img_128 = cv2.resize(img_sub, (128, 128))
    
    i = b.detect(img_128)
    l_face.append(i)



for iFace, faces in enumerate(l_face):

    if len (faces) > 0:
        print (faces)

        face = faces[0]

        bb = l_bb[iFace]
        img_sub = GetSubVolume2D(image, bb).copy()
        plt.imshow(img_sub)
        plt.show()

        draw_face_bb(img_sub, face)
        _draw_single_feature(img_sub, face, 'mouth', 2)
        _draw_single_feature(img_sub, face, 'r_eye', 2)
        _draw_single_feature(img_sub, face, 'l_eye', 2)
        _draw_single_feature(img_sub, face, 'nose', 2)
        _draw_single_feature(img_sub, face, 'r_ear', 2)
        _draw_single_feature(img_sub, face, 'l_ear', 2)

        plt.imshow(img_sub)
        plt.show()








# input image, face details


#################################################################################
#
#   blaze_after_mtcnn
#

def blaze_after_mtcnn(image, mtcnn_detector):

    width = image.shape[0]
    height = image.shape[1]

    features_mtcnn = mtcnn_detector.detect_faces(image)

    for face in features_mtcnn:

        x_min = face['box'][0]
        x_max = x_min + face['box'][2]

        y_min = face['box'][1]
        y_max = y_min + face['box'][3]

        mtcnn_confidence = face['confidence']


        






 

    








for i in range(num_frames):
    print(f"---------------------- Frame {i} -----------------------------")
    # blaze_after_mtcnn(video_real[i], mtcnn_detector)
    blaze_chunked(video_real[i], l_c, sample_size)


h = HaarCascade()

start = datetime.datetime.now()


for i in range(num_frames):
    print(f"---------------------- Frame {i} -----------------------------")
    # blaze_after_mtcnn(video_real[i], mtcnn_detector)
    faces = h.detect(video_real[i])

end   = datetime.datetime.now()

dT = (end - start).seconds

print(f"Processing time haar cascade: {dT}s")
    




l_diff = []

for i in range(num_frames):
    print(f"---------------------- Frame {i} -----------------------------")
    # blaze_after_mtcnn(video_real[i], mtcnn_detector)
    anDiff = np.array(get_region_diff(video_real[i], video_fake[i], l_c, sample_size))
    l_diff.append(anDiff)


anDiff = np.vstack(l_diff)

