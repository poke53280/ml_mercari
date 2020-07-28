
import gc
import os
import pathlib
import numpy as np
import pandas as pd
import sys
import keras
import tensorflow as tf

import cv2
from keras.models import load_model



isKaggle = pathlib.Path("/kaggle/input").is_dir()

if isKaggle:
    os.system('pip install /kaggle/input/mtcnnpackage/mtcnn-0.1.0-py3-none-any.whl')
    os.chdir('/kaggle/input/code-f2')

from mp4_frames import get_part_dir
from mp4_frames import get_test_dir
from mp4_frames import get_model_dir
from mp4_frames import get_submission_dir
from mp4_frames import read_video

from face_detector import MTCNNDetector
import pickle

from sklearn.metrics import mean_squared_error


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale



def get_linear_prediction_channel(t, x, isDraw):

    # transforming the data to include another axis
    t = t[:, np.newaxis]
    x = x[:, np.newaxis]def

    polynomial_features= PolynomialFeatures(degree=12)
    t_poly = polynomial_features.fit_transform(t)


    model = LinearRegression()
    model.fit(t_poly, x)

    x_pred = model.predict(t_poly)

    if isDraw:
        mse = mean_squared_error(x, x_pred)
        print(f"MSE: {mse}")

        plt.scatter(t, x, s=10)
        # sort the values of x before line plot
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(t,x_pred), key=sort_axis)
        t_out, x_poly_pred = zip(*sorted_zip)
        plt.plot(t_out, x_poly_pred, color='m')
        plt.show()

    return x_pred


####################################################################################
#
#   get_linear_prediction_row
#

def get_linear_prediction_row(t, data0, isDraw):
    px = get_linear_prediction_channel(t, data0[:, 0], isDraw).reshape(-1)
    py = get_linear_prediction_channel(t, data0[:, 1], isDraw).reshape(-1)
    pz = get_linear_prediction_channel(t, data0[:, 2], isDraw).reshape(-1)

    data0_ = np.array([px, py, pz]).T
    return data0_




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
#   find_spaced_out_faces_boxes
#

def find_spaced_out_faces_boxes(mtcnn_detector, video, nSpace):
    z_max = video.shape[0]

    l_key_frames = list(np.linspace(0, z_max - 1, endpoint = True, num = int(z_max/nSpace)).astype(np.int32))

    d = _get_face_boxes(mtcnn_detector, video, l_key_frames)

    return d


def find_all_face_boxes(mtcnn_detector, video):
    z_max = video.shape[0]

    l_key_frames = list(range(z_max))

    d = _get_face_boxes(mtcnn_detector, video, l_key_frames)

    return d


def _get_face_boxes(mtcnn_detector, video, l_key_frames):
    d = {}

    for iKeyFrame in l_key_frames:
        l_faces_key_frame = mtcnn_detector.detect(video[iKeyFrame])
        if len(l_faces_key_frame) > 0:
            d[iKeyFrame] = l_faces_key_frame

    return d





####################################################################################
#
#   get_random_face_box_from_z
#

def get_random_face_box_from_z(d, z, x_max, y_max, z_max):
    anKeys = np.array(list(d.keys()))

    assert z >= anKeys[0]
    assert z <= anKeys[-1]

    idx_key = np.abs(anKeys - z).argmin()

    iKeyFrame = anKeys[idx_key]

    l_faces = d[iKeyFrame]

    idx_face = np.random.choice(range(len(l_faces)))

    l_face = l_faces[idx_face]

    arBBMin = l_face['bb_min']
    arBBMax = l_face['bb_max']

    arScale = np.array([x_max, y_max])

    anBBMin = (arBBMin * arScale).astype(np.int32)
    anBBMax = (arBBMax * arScale).astype(np.int32)

    return (anBBMin, anBBMax)



####################################################################################
#
#   find_middle_face_box
#

def find_middle_face_box(mtcnn_detector, video):
    
    z_max = video.shape[0]
    y_max = video.shape[1]
    x_max = video.shape[2]

    l_faces = mtcnn_detector.detect(video[int(z_max/2)])

    l_bb_min = []
    l_bb_max = []
    l_confidence = []

    for x in l_faces:
        l_bb_min.append(x['bb_min'])
        l_bb_max.append(x['bb_max'])
        l_confidence.append(x['confidence'])

    
    anBBMin = np.array(l_bb_min)
    anBBMax = np.array(l_bb_max)
    anConfidence = np.array(l_confidence)

    iConfidenceMax = np.argmax(anConfidence)

    arBBMin = anBBMin[iConfidenceMax]
    arBBMax = anBBMax[iConfidenceMax]

    arScale = np.array([x_max, y_max])

    anBBMin = (arBBMin * arScale).astype(np.int32)
    anBBMax = (arBBMax * arScale).astype(np.int32)

    return (anBBMin, anBBMax)





####################################################################################
#
#   adjust_box_1d
#

def adjust_box_1d(c, half_size, extent):
    b_min = np.max([0, c - half_size])

    c = b_min + half_size

    b_max = np.min([extent, c + half_size])

    b_min = b_max - 2 * half_size

    assert (b_max - b_min) == 2 * half_size
    return (b_max + b_min) // 2


####################################################################################
#
#   cut_frame
#
# outputsize <= 0 : No resize

def cut_frame(bb_min, bb_max, orig_video, test_video, z_sample, outputsize, isDraw):

    rDiagnonal = (bb_max - bb_min) * (bb_max - bb_min)

    n_characteristic_face_size = np.sqrt(rDiagnonal[0] + rDiagnonal[1]).astype(np.int32)

    image_real = orig_video[z_sample].copy()
    image_test = test_video[z_sample].copy()

    x_max = image_real.shape[1]
    y_max = image_real.shape[0]

    sample_size = int(0.5 * n_characteristic_face_size)
    half_size = int (sample_size/2)

    center = 0.5 * (bb_min + bb_max)
    center = center.astype(np.int32)

    center_adjusted = np.array([adjust_box_1d(center[0], half_size, x_max), adjust_box_1d(center[1], half_size, y_max)])

    s_min = center_adjusted - half_size
    s_max = center_adjusted + half_size

    real_sample = image_real[s_min[1]:s_max[1], s_min[0]:s_max[0]].copy()
                
    test_sample = image_test[s_min[1]:s_max[1], s_min[0]:s_max[0]].copy()
               
    if outputsize > 0:
        test_sample = cv2.resize(test_sample, (outputsize, outputsize))
        real_sample = cv2.resize(real_sample, (outputsize, outputsize))

    return (real_sample, test_sample)


####################################################################################
#
#   sample
#

def sample(test_image, n_target_size):
    width = test_image.shape[1]
    height = test_image.shape[0]

    h0 = np.random.choice(height)
    h1 = np.random.choice(height)

    p0 = (0,h0)
    p1 = (width-1, h1)

    p0 = (*p0, 0)
    p1 = (*p1, 0)

    l = get_line(np.array(p0), np.array(p1))
    
    l = l[0:2]

    l = l.astype(np.int32)

    n_border = int ((l.shape[1] - n_target_size) / 2)

    assert n_border >= 0, "n_border >= 0"

    l = l[:, n_border:n_border + n_target_size]

    assert l.shape[1] == n_target_size, "l.shape[1] == n_target_size"

    test_line = test_image[l[1], l[0]]

    rAngle = np.arctan2( (h1 - h0), (width - 1))
    rPix = np.sqrt(width * width + height * height)

    return rAngle, rPix, test_line


####################################################################################
#
#   sample_video_predict
#


def sample_video_predict(mtcnn_detector, path):

    l_line = []
    l_fake = []
    l_angle = []
    l_pix = []

    n_target_size = 100
    outputsize = 128 + 64
    orig_video = read_video(path, 0)

    z_max = orig_video.shape[0]
    y_max = orig_video.shape[1]
    x_max = orig_video.shape[2]


    d_faces = _get_face_boxes(mtcnn_detector, orig_video, [0])

    bb_min, bb_max = get_random_face_box_from_z(d_faces, 0, x_max, y_max, z_max)

    real_image, _ = cut_frame(bb_min, bb_max, orig_video, orig_video, 0, outputsize, False)

    for i in range(500):
        rAngle, rPix, test_line = sample(real_image, n_target_size)

        l_line.append(test_line)
        l_fake.append(True)
        l_angle.append(rAngle)
        l_pix.append(rPix)  

    df = pd.DataFrame({'fake': l_fake, 'angle': l_angle, 'pix': l_pix, 'data': l_line})
    
    return df




####################################################################################
#
#   preprocess_input
#


def preprocess_input(data):

    t = np.arange(0, data.shape[1])

    l_x = []
    for x in data:
        l_x.append(get_linear_prediction_row(t, x, False))

    data_p = np.stack(l_x)

    data_ = (data - data_p)

    data0_ = data_[:, :, 0]
    data1_ = data_[:, :, 1]
    data2_ = data_[:, :, 2]

    data0_s = scale(data0_, axis = 1)
    data1_s = scale(data1_, axis = 1)
    data2_s = scale(data2_, axis = 1)

    data_ = np.stack([data0_s, data1_s, data2_s], axis = 2)

    return data_



def predict(model, data):
    data = data.reshape(-1, 300)

    data_p = model.predict(data[:200])
    mse = mean_squared_error(data[:200].reshape(-1), data_p.reshape(-1))

    return mse




def load_model_pair(model_cluster, model_name):

    real_file = get_model_dir() / f"c_{model_cluster}_{model_name}_real.h5"
    fake_file = get_model_dir() / f"c_{model_cluster}_{model_name}_fake.h5"
    assert real_file.is_file() and fake_file.is_file()

    model_real = load_model(real_file)
    model_fake = load_model(fake_file)

    return model_real, model_fake



def predict_single_test_file(x):

    df = sample_video_predict(mtcnn_detector, x)

    data = np.stack(df.data.values)
    data = preprocess_input(data)

    stat = []

    for m in l_models:

        model_real = m[0]
        model_fake = m[1]

        err_mr = predict(model_real,  data)
        err_mf = predict(model_fake,  data)

        stat.append(err_mr)
        stat.append(err_mf)

    acReal = np.array(stat)
    acReal0 = acReal[::2]
    acReal1 = acReal[1::2]

    acDiffReal = acReal0 - acReal1

    return acDiffReal

if isKaggle:
    os.chdir('/kaggle/working')

l_m = [('200', 'qhhkcsvlod'), ('201', 'ahkibiituu'), ('201', 'ajconjiwey'), ('210', 'dfembozird'), ('210', 'lhtohlvehk'), ('210', 'yrqhcjnpix'), ('211', 'copowfosob'), ('211', 'ctlqptsltq'), ('211', 'ddqybqgnkl'), ('220', 'aguxjvffln'), ('220', 'akmkangqbj'), ('220', 'aqtypfezoi'), ('220', 'biotzvraxy'), ('220', 'bthweewuqp'), ('220', 'bwvmskoriy'), ('220', 'cyzgavhyiv')]


l_models = []

for x in l_m:
    model_real, model_fake = load_model_pair(x[0], x[1])
    l_models.append((model_real, model_fake))


model_stage2 = pickle.load(open(get_model_dir() / "finalized_model.sav", 'rb'))


input_dir = get_test_dir()

model_dir = get_model_dir()
submission_dir = get_submission_dir()


mtcnn_detector = MTCNNDetector()

l_files = list (sorted(input_dir.iterdir()))

l_filenames = [str(x.name) for x in l_files]

d_res = {}

for ix, x in enumerate(l_files[:3]):
    gc.collect()
    print(x)
    #try:
    stage1_pred = predict_single_test_file(x)
    stage2_pred = model_stage2.predict(stage1_pred.reshape(1, -1))

    if stage2_pred == True:
        d_res[str(x.name)] = 0.0
    else:
        d_res[str(x.name)] = 1.0
    #except:
    #    print("Unexpected error:", sys.exc_info()[0])
    #    d_res[str(x.name)] = 0.5
    
"""c"""


sub = pd.DataFrame()

sub['filename'] = l_filenames

sRes = sub.filename.map(d_res)

sRes = sRes.fillna(0.5)

sub = sub.assign(label = sRes)

sub.to_csv(get_submission_dir() / 'submission.csv',index=False)

print("All done.")



