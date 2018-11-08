
import numpy as np
from datetime import datetime
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import gc



def generate_single_target(df_meta_all, df_all, itarget, num_desired_elements):

    m_meta = (df_meta_all.target == itarget)

    objs = np.unique(df_meta_all[m_meta].object_id)

    m_data = df_all.object_id.isin(objs)

    df_meta = df_meta_all[m_meta].copy()
    df = df_all[m_data].copy()

    num_objects = df_meta.shape[0]

    num_sets = 1 + int (num_desired_elements / num_objects)

    num_elements = num_sets * num_objects

    d0, y_0 = generate_sets(df_meta, df, num_sets)

    assert d0.shape[0] == num_elements

    return d0, y_0

"""c"""


DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

df_meta_all = pd.read_csv(DATA_DIR + "training_set_metadata.csv")
df_all = pd.read_csv(DATA_DIR + "training_set.csv")



d0, y_0 = generate_single_target(df_meta_all, df_all, 52, 500000)

d1, y_1 = generate_single_target(df_meta_all, df_all, 90, 500000)


data = np.vstack([d0, d1])
y = np.hstack([y_0, y_1])








