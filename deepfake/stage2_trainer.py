

import numpy as np
import pandas as pd
import pathlib
from keras.models import load_model
from sklearn.metrics import mean_squared_error

from mp4_frames import get_ready_data_dir
from mp4_frames import get_model_dir

from featureline import get_feature_converter

from dae_lstm import preprocess_input

from featureline import sample_single

import lightgbm as lgbm


####################################################################################
#
#   get_accumulated_stats
#

def get_accumulated_stats(lines_in, lines_out):

    d_acc = {}

    d_acc['mse'] = mean_squared_error(lines_in.reshape(-1), lines_out.reshape(-1))

    lines_diff = lines_out - lines_in

    lines_diff = lines_diff.reshape(-1, 3 * 16)

    lines_max = np.max(lines_diff, axis = 1)
    lines_min = np.min(lines_diff, axis = 1)
    

    # max
    d_acc['acc_max_mean'] = np.mean(lines_max)
    d_acc['acc_max_var'] = np.var(lines_max)
    d_acc['acc_max_99']  = np.quantile(lines_max, 0.99)
    d_acc['acc_max_01'] = np.quantile(lines_max, 0.01)

    # min
    d_acc['acc_min_mean'] = np.mean(lines_min)
    d_acc['acc_min_var'] = np.var(lines_min)
    d_acc['acc_min_99']  = np.quantile(lines_min, 0.99)
    d_acc['acc_min_01'] = np.quantile(lines_min, 0.01)

    # argmax
    lines_argmax = np.argmax(lines_diff, axis = 1)
    d_acc['acc_argmax_mean'] = np.mean(lines_argmax)
    d_acc['acc_argmax_var'] = np.var(lines_argmax)
    d_acc['acc_argmax_99']  = np.quantile(lines_argmax, 0.99)
    d_acc['acc_argmax_01'] = np.quantile(lines_argmax, 0.01)

    # argmin
    lines_argmin = np.argmin(lines_diff, axis = 1)
    d_acc['acc_argmin_mean'] = np.mean(lines_argmin)
    d_acc['acc_argmin_var'] = np.var(lines_argmin)
    d_acc['acc_argmin_99']  = np.quantile(lines_argmin, 0.99)
    d_acc['acc_argmin_01'] = np.quantile(lines_argmin, 0.01)

    return d_acc


####################################################################################
#
#   get_accumulated_stats_init
#

def get_accumulated_stats_init():
    d_acc = {}

    d_acc['mse'] = -1
    

    # max
    d_acc['acc_max_mean'] = 0
    d_acc['acc_max_var'] = 0
    d_acc['acc_max_99']  = 0
    d_acc['acc_max_01'] = 0

    # min
    d_acc['acc_min_mean'] = 0
    d_acc['acc_min_var'] = 0
    d_acc['acc_min_99']  = 0
    d_acc['acc_min_01'] = 0

    # argmax
    d_acc['acc_argmax_mean'] = 0
    d_acc['acc_argmax_var'] = 0
    d_acc['acc_argmax_99']  = 0
    d_acc['acc_argmax_01'] = 0

    # argmin
    d_acc['acc_argmin_mean'] = 0
    d_acc['acc_argmin_var'] = 0
    d_acc['acc_argmin_99']  = 0
    d_acc['acc_argmin_01'] = 0

    return d_acc


####################################################################################
#
#   predict_single_file
#

def predict_single_file(m, x, isVerbose):

    d_f = get_feature_converter()

    if x.is_file():
        pass
    else:
        print(f"Not a file: {str(x)}")
        return get_accumulated_stats_init()

    try:
        data = sample_single(x)
    except Exception as err:
        print(err)
        data = None

    if data is None:
        return get_accumulated_stats_init()

    anFeature = data[:, 0]

    data = data[:, 1:]

    data = data.reshape(-1, 16, 3)

    if isVerbose:
        print (data[0])

    num_rows = data.shape[0]
    assert num_rows % len (d_f.keys()) == 0

    zFeature = 'l_mouth'
    iF = d_f[zFeature]
    m_correct_feature = (anFeature == iF)

    lines_in = preprocess_input(data[m_correct_feature])

    try:
        lines_out = m.predict(lines_in)
    except Exception as err:
        print(err)
        return get_accumulated_stats_init()

    if isVerbose:
        print (lines_out[0])

    d_acc = get_accumulated_stats(lines_in, lines_out)

    if isVerbose:
        print (d_acc)

    return d_acc


####################################################################################
#
#   train_stage2
#

def train_stage2():


    model_dir = get_model_dir()

    m = load_model(get_model_dir() / "my_model_l_mouth_rr.h5")


    input_dir = get_ready_data_dir()

    data = np.load(input_dir / "test_l_mouth.npy")
    df_meta = pd.read_pickle(input_dir / "test_meta.pkl")

    data = preprocess_input(data.reshape(-1, 16, 3))

    data_out = m.predict(data, verbose = 1, batch_size = 256)


    np.save(input_dir / "test_l_mouth_reconstruction.npy", data_out)

    data_out = np.load(input_dir / "test_l_mouth_reconstruction.npy")


    ##################### df meta preprocessing ################


    zVideo = df_meta['iPart'].astype('str') + df_meta['video']

    azVideo = np.array(zVideo)

    azVideoUnique = np.unique(azVideo)

    aiVideo = np.searchsorted(azVideoUnique, azVideo)

    df_meta = df_meta.assign(id = aiVideo)

    df_meta = df_meta.drop(['iPart', 'video'], axis = 1)


    aID = np.unique(df_meta.id)

    l = []

    for id in aID:

        m_id = df_meta.id == id

        y = df_meta[m_id].iloc[0].y

        lines_in = data[m_id]
        lines_out = data_out[m_id]

        d_acc = get_accumulated_stats(lines_in, lines_out)
        d_acc['y'] = y
        l.append(d_acc)

    df = pd.DataFrame(l)

    num_rows = df.shape[0]
    num_train = int (0.9 * num_rows)


    x_cols = [x for x in list (df.columns) if x != 'y']

    X_train = df[x_cols][:num_train]
    X_test = df[x_cols][num_train:]

    y = df.y.copy()

    m_fake = (y == 'fake')
    m_real = (y == 'real')

    y[m_fake] = '1'
    y[m_real] = '0'

    y = y.astype(np.int)

    y_train = y[:num_train]
    y_test = y[num_train:]


    params = {
        'objective' :'binary',
        'learning_rate' : 0.01,
        'num_leaves' : 3,
        'feature_fraction': 0.64, 
        'bagging_fraction': 0.8, 
        'bagging_freq':1,
        'boosting_type' : 'gbdt',
        'metric': 'binary_logloss'
    }

    # making lgbm datasets for train and valid
    d_train = lgbm.Dataset(X_train, y_train)
    d_valid = lgbm.Dataset(X_test, y_test)
    
    m2 = lgbm.train(params, d_train, 30, valid_sets=[d_train, d_valid], verbose_eval=1)

    m2.save_model(str(get_model_dir() / 'm2.txt'))






