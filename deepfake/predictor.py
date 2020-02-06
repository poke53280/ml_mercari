
import numpy as np
import pandas as pd
import pathlib

from shutil import copyfile

isKaggle = pathlib.Path("/kaggle/input").is_dir()

# !pip install /kaggle/input/mtcnnpackage/mtcnn-0.1.0-py3-none-any.whl
# Bytt til os install.

if isKaggle:

    copyfile(src = "../input/pythoncode/dae_lstm.py", dst = "../working/dae_lstm.py")

    copyfile(src = "../input/pythoncode/easy_face.py", dst = "../working/easy_face.py")
    copyfile(src = "../input/pythoncode/face_detector.py", dst = "../working/face_detector.py")
    copyfile(src = "../input/pythoncode/feature_trainer.py", dst = "../working/feature_trainer.py")
    copyfile(src = "../input/pythoncode/image_grid.py", dst = "../working/image_grid.py")
    copyfile(src = "../input/pythoncode/line_sampler.py", dst = "../working/line_sampler.py")
    copyfile(src = "../input/pythoncode/mp4_frames.py", dst = "../working/mp4_frames.py")
    copyfile(src = "../input/pythoncode/stage2_trainer.py", dst = "../working/stage2_trainer.py")
    copyfile(src = "../input/pythoncode/featureline.py", dst = "../working/featureline.py")



from keras.models import load_model
from lightgbm import Booster
from lightgbm import Dataset

from mp4_frames import get_test_dir
from mp4_frames import get_model_dir
from mp4_frames import get_submission_dir

from dae_lstm import preprocess_input
from dae_lstm import reconstruction_error
from stage2_trainer import predict_single_file



input_dir = get_test_dir()
model_dir = get_model_dir()
submission_dir = get_submission_dir()

l_files = list (input_dir.iterdir())

m1 = load_model(get_model_dir() / "my_model_l_mouth_rr.h5")

m2 = Booster(model_file = str(get_model_dir() / "m2.txt"))

d_res = {}

for i, x in enumerate(l_files):
    print (i)

    d_res[str(x.name)] = predict_single_file(m1, x)

"""c"""

df = pd.DataFrame(d_res).T

df = df.reset_index()
df = df.rename(columns = {'index' : 'filename'})

df = df.assign(label = 0.5)

label = df.label.copy()

m_invalid = (df.mse == 0) & (df.acc_argmax_99 == 0) & (df.acc_argmax_01 == 0)

x_cols = [x for x in list (df.columns) if (x != "filename") and (x != "label")]

label[~m_invalid] = m2.predict(df[x_cols][~m_invalid])

df = df.assign(label = label)

df[['filename', 'label']].to_csv(submission_dir / 'submission.csv',index=False)











