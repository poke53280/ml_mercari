
import gc
import numpy as np
import pandas as pd
import pathlib
from keras.models import load_model
from sklearn.metrics import mean_squared_error

from mp4_frames import get_ready_data_dir
from mp4_frames import get_model_dir
from mp4_frames import get_pred0_dir

from featureline import get_feature_converter

from dae_lstm import preprocess_input

from featureline import sample_single
from featureline import get_error_line
from featureline import is_error_line

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
#   predict_stage1_single_file
#

def predict_stage1_single_file(mtcnn_detector, m1, x, isVerbose):

    d_f = get_feature_converter()

    if x.is_file():
        pass
    else:
        print(f"Not a file: {str(x)}")
        return get_accumulated_stats_init()

    try:
        data = sample_single(mtcnn_detector, x, 0.3)
    except Exception as err:
        print(err)
        data = get_error_line()

    if is_error_line(data):
        return get_accumulated_stats_init()

    anFeature = data[:, 0]

    data = data[:, 1:]

    data = data.reshape(-1, 32, 3)

    if isVerbose:
        print (data[0])

    num_rows = data.shape[0]
    assert num_rows % len (d_f.keys()) == 0

    zFeature = 'l_mouth'
    iF = d_f[zFeature]
    m_correct_feature = (anFeature == iF)

    lines_in = preprocess_input(data[m_correct_feature])

    del data
    gc.collect()

    # Todo check out (possible need for) predict on batch

    try:
        lines_out = m1.predict(lines_in)
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
#   predict_stage1
#
#
# e.g. predict_stage1("l_mouth", "rr", 0, 1)
#

def predict_stage1(zFeature, zModel_type, iPartMin, iPartMax):

    model_dir = get_model_dir()
    input_dir = get_ready_data_dir()
    output_dir = get_pred0_dir()


    modelfile = get_model_dir() / f"model_{zFeature}_{zModel_type}.h5"
    assert modelfile.is_file()

    data_file = input_dir / f"test_{zFeature}_p_{iPartMin}_p_{iPartMax}.npy"
    assert data_file.is_file()

    meta_file = input_dir / f"test_meta_p_{iPartMin}_p_{iPartMax}.pkl"
    assert meta_file.is_file()

    output_file = output_dir / f"pred_{zFeature}_p_{iPartMin}_p_{iPartMax}_{zModel_type}.npy"

    m = load_model(modelfile)
    data = np.load(data_file)
    df_meta = pd.read_pickle(meta_file)

    assert data.shape[0] == df_meta.shape[0]

    data = preprocess_input(data.reshape(-1, 32, 3))

    data_out = m.predict(data, verbose = 1, batch_size = 256)

    rms = mean_squared_error(data_out.reshape(-1), data.reshape(-1))

    print (f"RMS = {rms}")

    np.save(output_file, data_out)


####################################################################################
#
#   train_stage2
#

# e.g. train_stage2("l_mouth", "rr", 0, 1)

def train_stage2(zFeature, zModel_type, iPartMin, iPartMax):


    input_dir = get_ready_data_dir()
    pred_dir = get_pred0_dir()

    data_file = input_dir / f"test_{zFeature}_p_{iPartMin}_p_{iPartMax}.npy"
    assert data_file.is_file()

    meta_file = input_dir / f"test_meta_p_{iPartMin}_p_{iPartMax}.pkl"
    assert meta_file.is_file()

    prediction_file = output_dir / f"pred_{zFeature}_p_{iPartMin}_p_{iPartMax}_{zModel_type}.npy"
    assert prediction_file.is_file()

    m2_file = model_dir / f"m2_{zFeature}_p_{iPartMin}_p_{iPartMax}_{zModel_type}.txt"


    data = np.load(data_file)
    data = preprocess_input(data.reshape(-1, 32, 3))

    df_meta = pd.read_pickle(meta_file)

    data_p = np.load(prediction_file)

    assert data.shape[0] == df_meta.shape[0]
    assert data.shape == data_p.shape

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
        lines_out = data_p[m_id]

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
        'num_leaves' : 5,
        'feature_fraction': 0.64, 
        'bagging_fraction': 0.8, 
        'bagging_freq':1,
        'boosting_type' : 'gbdt',
        'metric': 'binary_logloss'
    }

    # making lgbm datasets for train and valid
    d_train = lgbm.Dataset(X_train, y_train)
    d_valid = lgbm.Dataset(X_test, y_test)
    
    m2 = lgbm.train(params, d_train, 290, valid_sets=[d_train, d_valid], verbose_eval=1)

    _ = m2.save_model(str(m2_file))









