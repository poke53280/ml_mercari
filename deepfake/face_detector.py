

import numpy as np
import pandas as pd
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from blazeface import BlazeFace

from mtcnn.mtcnn import MTCNN

from mp4_frames import get_part_dir

from mp4_frames import read_metadata
from mp4_frames import read_video

from image_grid import GetGrid2DBB
from image_grid import GetSubVolume2D
from line_sampler import get_random_trace_lines
from line_sampler import sample_single_cube


import matplotlib.pyplot as plt
import datetime

import cv2

import random


######################################################################
#
#   get_test_video
#

def get_test_video(num_frames):

    l_part = [0, 6, 7, 10, 18, 28, 37, 49]

    iPart = random.choice(l_part)

    video_dir = get_part_dir(iPart)
    l_d = read_metadata(iPart)

    idx = np.random.choice(len (l_d))

    current = l_d[idx][0]

    filepath = video_dir / current

    video = read_video(filepath, num_frames)

    return video



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
#   _draw_face_bb
#

def _draw_face_bb(image, face):

    x_shape = image.shape[1]
    y_shape = image.shape[0]

    x0 = int (face['bb_min'][0] * x_shape)
    y0 = int (face['bb_min'][1] * y_shape)

    x1 = int (face['bb_max'][0] * x_shape)
    y1 = int (face['bb_max'][1] * y_shape)

    print(f"({x0}, {y0}) - ({x1}, {y1})")

    image = cv2.rectangle(image, (x0, y0), (x1, y1), (255,0,0), 5)

######################################################################
#
#   _cut_face_bb
#

def _cut_face_bb(image, face):

    x_shape = image.shape[1]
    y_shape = image.shape[0]

    x0 = int (face['bb_min'][0] * x_shape)
    y0 = int (face['bb_min'][1] * y_shape)

    x1 = int (face['bb_max'][0] * x_shape)
    y1 = int (face['bb_max'][1] * y_shape)

    print(f"({x0}, {y0}) - ({x1}, {y1})")

    sub_image = image[y0:y1, x0:x1, :].copy()

    return sub_image

######################################################################
#
#   _fit_1D
#

def _fit_1D(c, half_size, extent):

    c = int (c * extent)

    c0 = c - half_size
    c1 = c + half_size

    if c0 < 0:
        c0 = 0
        c1 = (half_size * 2)

    if c1 > extent:
        c1 = extent
        c0 = ce1 - half_size

    assert (c1 - c0) == (half_size * 2)       

    return (c0, c1)



######################################################################
#
#   _draw_single_feature
#

def _draw_single_feature(image, face, zFeature, rect_size):

    assert rect_size > 0

    x_shape = image.shape[1]
    y_shape = image.shape[0]

    x = int (face[zFeature][0] * x_shape)
    y = int (face[zFeature][1] * y_shape)

    x0 = x - rect_size
    y0 = y - rect_size

    x1 = x + rect_size
    y1 = y + rect_size

    image = cv2.rectangle(image, (x0, y0), (x1, y1), (255,0,0), 2)

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

        self.detections = []


    def detect(self, img):
        assert img.shape == (128, 128, 3)
        detection = self.net.predict_on_image(img)

        l_face = []
        
        for anFace in detection:

            face  = {}

            face['bb_min']  = (anFace[1].item(), anFace[0].item())
            face['bb_max']  = (anFace[3].item(), anFace[2].item())

            face['r_eye']   = (anFace[4].item(), anFace[5].item())
            face['l_eye']   = (anFace[6].item(), anFace[7].item())

            face['c_nose']    = (anFace[8].item(), anFace[9].item())
            face['c_mouth']   = (anFace[10], anFace[11])

            face['r_ear']   = (anFace[12].item(), anFace[13].item())        
            face['l_ear']   = (anFace[14].item(), anFace[15].item())  

            face['confidence'] = anFace[16].item()
            
            l_face.append(face)
        return l_face


    def draw(self, image, l_face):
            x_shape = image.shape[1]
            y_shape = image.shape[0]

            for face in l_face:
                _draw_face_bb(image, face)
                _draw_single_feature(image, face, 'c_mouth', 2)
                _draw_single_feature(image, face, 'r_eye', 2)
                _draw_single_feature(image, face, 'l_eye', 2)
                _draw_single_feature(image, face, 'c_nose', 2)
                _draw_single_feature(image, face, 'r_ear', 2)
                _draw_single_feature(image, face, 'l_ear', 2)

    

######################################################################
#
#   MTCNNDetector
#

class MTCNNDetector():

    def __init__(self):
         self.detector = MTCNN()

    def _get_face_from_mtcnn(self, face_mtcnn, heigth, width):
    
        d_face = {}

        # Face bounding box
    
        x0 = face_mtcnn['box'][0]
        x1 = x0 + face_mtcnn['box'][2]

        x0 = x0 / width
        x1 = x1 / width

        y0 = face_mtcnn['box'][1]
        y1 = y0 + face_mtcnn['box'][3]

        y0 = y0 / heigth
        y1 = y1 / heigth

        d_face['bb_min']  = (x0, y0)
        d_face['bb_max']  = (x1, y1)

        d_face['confidence'] = face_mtcnn['confidence']

        keypoints = face_mtcnn['keypoints']

        d_face['l_eye'] = keypoints['left_eye']
        d_face['r_eye'] = keypoints['right_eye']
        d_face['c_nose'] = keypoints['nose']
        d_face['l_mouth'] = keypoints['mouth_left']
        d_face['r_mouth'] = keypoints['mouth_right']

        for x in ['l_eye', 'r_eye', 'c_nose',  'l_mouth', 'r_mouth']:
            d_face[x] = (d_face[x][0]/ width, d_face[x][1]/ heigth)

        d_face = self.get_face_area(d_face)

        return d_face

    def detect(self, image):

        l_face = []

        heigth = image.shape[0]
        width = image.shape[1]

        features_mtcnn = self.detector.detect_faces(image)

        for x in features_mtcnn:
            face = self._get_face_from_mtcnn(x, heigth, width)
            l_face.append(face)

        return l_face
    

    def draw(self, image, l_face):
        x_shape = image.shape[1]
        y_shape = image.shape[0]

        for face in l_face:
            _draw_face_bb(image, face)
            _draw_single_feature(image, face, 'r_eye', 2)
            _draw_single_feature(image, face, 'l_eye', 2)
            _draw_single_feature(image, face, 'c_nose', 2)
            _draw_single_feature(image, face, 'l_mouth', 2)
            _draw_single_feature(image, face, 'r_mouth', 2)

    def get_face_area(self, face):
        l_xFeatures = [  face['c_nose'][0],  face['l_eye'][0], face['r_eye'][0], face['l_mouth'][0], face['r_mouth'][0]]
        l_yFeatures = [  face['c_nose'][1],  face['l_eye'][1], face['r_eye'][1], face['l_mouth'][1], face['r_mouth'][1]]

        x0 = np.min(l_xFeatures)
        x1 = np.max(l_xFeatures)

        y0 = np.min(l_yFeatures)
        y1 = np.max(l_yFeatures)

        face['f_min'] = (x0, y0)
        face['f_max'] = (x1, y1)

        return face

       
        


       
weight_path = pathlib.Path(f"C:\\Users\\T149900\\Documents\\GitHub\\ml_mercari\\deepfake")
assert weight_path.is_dir()


######################################################################
#
#   sample_video
#

def sample_video(video, video_dept, cube_size, isDraw):

    assert video.shape[0] == video_dept

    iFrame = video.shape[0]//2

    image = video[iFrame]

    x_shape = image.shape[1]
    y_shape = image.shape[0]

    if isDraw:
        plt.imshow(image)
        plt.show()

    mtcnn = MTCNNDetector()

    t0 = datetime.datetime.now()

    l_faces = mtcnn.detect(image)

    t1 = datetime.datetime.now()

    for x in l_faces:
        print(f"Confidence {x['confidence']}")

    dt_mtcnn  = (t1 - t0).total_seconds()

    print(f"mtcnn deetect time: {dt_mtcnn}s")

    if isDraw:
        image_mtcnn_rect = image.copy()
        mtcnn.draw(image_mtcnn_rect, l_faces)
        plt.imshow(image_mtcnn_rect)
        plt.show()
    

    l_video_cube = []

    for x in l_faces:

        x0 = x['f_min'][0]
        y0 = x['f_min'][1]

        x1 = x['f_max'][0]
        y1 = x['f_max'][1]

        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)

        x_size = (x1 - x0) * x_shape
        y_size = (y1 - y0) * y_shape

        face_size = int ((1.8 * np.min([x_size, y_size]) // 2) * 2)

        print(f"Characteristic face size {face_size} vs sample size {cube_size}")

        (x0, x1) = _fit_1D(cx, cube_size // 2, x_shape)
        (y0, y1) = _fit_1D(cy, cube_size // 2, y_shape)

        if isDraw:
            img_sub = image[y0:y1, x0:x1, :]
            plt.imshow(img_sub)
            plt.show()

        video_sub = video[:, y0:y1, x0:x1, :]

        assert video_sub.shape == (video_dept, cube_size, cube_size, 3)

        l_video_cube.append(video_sub)

    return l_video_cube





######################################################################
#
#   sample_video_outer
#

def sample_video_outer(video_base):

    l_video_sub = sample_video(video_base, 32, 64, False)

    l_samples = []
    for vid_sub in l_video_sub:
        c00 = vid_sub[:, :32, :32, :]
        c10 = vid_sub[:, 32:, :32, :]
        c01 = vid_sub[:, :32, 32:, :]
        c11 = vid_sub[:, 32:, 32:, :]

        num_samples = 1000

        l_samples.append(sample_single_cube(c00, get_random_trace_lines(num_samples)))
        l_samples.append(sample_single_cube(c10, get_random_trace_lines(num_samples)))
        l_samples.append(sample_single_cube(c01, get_random_trace_lines(num_samples)))
        l_samples.append(sample_single_cube(c11, get_random_trace_lines(num_samples)))

    anSamples = np.concatenate(l_samples)

    #idxSamples = np.random.choice(anSamples.shape[0], 4000, replace = False)

    #anSamples = anSamples[idxSamples]

    return anSamples



t0 = datetime.datetime.now()
video_base = get_test_video(32)
t1 = datetime.datetime.now()
dt_read = (t1 - t0).total_seconds()
print(f"Video read time: {dt_read}s")

aS = sample_video_outer(video_base)

t2 = datetime.datetime.now()

dt_all = (t2 - t0).total_seconds()

print(f"Total read and sample time: {dt_all}s")





b = BlazeDetector(weight_path)


sample_size = 800
rOverlap = 1.3

l_bb = GetGrid2DBB(image.shape[0], image.shape[1], sample_size, 1.3)

l_face = []

t0 = datetime.datetime.now()

for bb in l_bb:
    
    img_sub = GetSubVolume2D(image, bb)
    img_128 = cv2.resize(img_sub, (128, 128))
    
    i = b.detect(img_128)
    l_face.append(i)


t1 = datetime.datetime.now()

dt_blaze_chunked = (t1 - t0).total_seconds()

print(f"Blaze detect time: {dt_blaze_chunked}s")


for iFace, faces in enumerate(l_face):

    if len (faces) > 0:
        print (faces)

        bb = l_bb[iFace]
        img_sub = GetSubVolume2D(image, bb).copy()
        plt.imshow(img_sub)
        plt.show()

        b.draw(img_sub, faces)

        plt.imshow(img_sub)
        plt.show()







