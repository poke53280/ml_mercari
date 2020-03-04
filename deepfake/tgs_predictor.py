

import numpy as np
import pandas as pd

from mp4_frames import get_meta_dir

from keras.models import load_model

from tgs_model_funcs import bce_dice_loss

from tgs_model_funcs import my_iou_metric

from tgs_model_funcs import load_to_series_rgb
from tgs_model_funcs import load_to_series_grayscale

import matplotlib.pyplot as plt

img_size_target = 128
img_size_ori = 128


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    #return res



model = load_model(get_meta_dir() / "model_2", custom_objects = {'bce_dice_loss' : bce_dice_loss, 'my_iou_metric': my_iou_metric})

df = pd.read_pickle(get_meta_dir() / "df_tgs.pkl")

df = df[df.m_test]

m_fake = (df.original != df.file_stem)

df = df.assign(fake = m_fake)


g = df.groupby('file_stem')

s_files = g.file.apply(list)
s_masks = g.file_mask.apply(list)
s_target = g.fake.first()

df_pred = pd.DataFrame({'l_files': s_files, 'l_masks': s_masks, 'y' : s_target})

for idx in range(df_pred.shape[0]):

    test_video = df_pred.iloc[idx]

    l_files = test_video.l_files
    l_masks = test_video.l_masks
    y = test_video.y

    sValid, sData = load_to_series_rgb(l_files)
    sValidMask, sMask = load_to_series_grayscale(l_masks)

    anData = np.array(sData.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3)
    anMask = np.array(sMask.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

    sCoverage = sMask.map(np.sum) / pow(img_size_ori, 2)

    anPredict = model.predict(anData)

    anPredict = anPredict.reshape(anData.shape[0], -1)

    print(f"Fake: {y}. Mean max: {np.mean(anPredict, axis = 1).max()}")

   






