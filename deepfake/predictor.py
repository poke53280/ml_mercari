
import os
import pathlib
import numpy as np
import pandas as pd

import keras
import tensorflow as tf

from keras.models import load_model
from lightgbm import Booster
from lightgbm import Dataset


isKaggle = pathlib.Path("/kaggle/input").is_dir()

if isKaggle:
    os.system('pip install /kaggle/input/mtcnnpackage/mtcnn-0.1.0-py3-none-any.whl')
    os.chdir('/kaggle/input/pythoncode')



from mp4_frames import get_part_dir

from mp4_frames import get_test_dir
from mp4_frames import get_model_dir
from mp4_frames import get_submission_dir

from dae_lstm import preprocess_input
from dae_lstm import reconstruction_error
from stage2_trainer import predict_single_file
from stage2_trainer import get_accumulated_stats_init

from mtcnn.mtcnn import MTCNN


print(f"Tensorflow {tf.__version__}")
print(f"Keras {keras.__version__}")



if isKaggle:
    os.chdir('/kaggle/working')

input_dir = get_test_dir()

#input_dir = get_part_dir(0)


model_dir = get_model_dir()
submission_dir = get_submission_dir()

m1 = load_model(get_model_dir() / "my_model_l_mouth_rr.h5")
m2 = Booster(model_file = str(get_model_dir() / "m2.txt"))

l_files = list (sorted(input_dir.iterdir()))

l_filenames = [str(x.name) for x in l_files]

num_files = 3000


if num_files == 0:
    num_files = len (l_files)


d_res = {}


for i, x in enumerate(l_files[:num_files]):
    print (i)

    try:
        d_res[str(x.name)] = predict_single_file(m1, x, i < 3)
    except Exception as err:
        d_res[str(x.name)] = get_accumulated_stats_init()
    
"""c"""

print("Done stage 1 prediction")

df = pd.DataFrame(d_res).T

df = df.reset_index()
df = df.rename(columns = {'index' : 'filename'})

m_invalid = (df.mse < 0)
m_valid = ~m_invalid

x_cols = [x for x in list (df.columns) if (x != "filename") and (x != "label")]

print("Starting stage 2 prediction")

label = m2.predict(df[x_cols])

label[m_invalid] = 0.5

assert (label >= 0).all()

df = df.assign(label = label)

print("Done stage 2 prediction")

sub=pd.DataFrame()

sub['filename'] = l_filenames
sub['label_sub'] = 0.5

sub = pd.merge(left = sub, right = df[['label','filename']], on = 'filename', how = 'left')

m_got_value = ~sub['label'].isna()

label = sub.label_sub.copy()

label[m_got_value] = sub['label'][m_got_value]

sub = sub.assign(label = label)
sub = sub.drop('label_sub', axis = 1)

sub.to_csv(get_submission_dir() / 'submission.csv',index=False)

print("All done.")



