

import numpy as np
from mp4_frames import get_output_dir
from mp4_frames import get_ready_data_dir
from featureline import get_feature_converter

import pandas as pd


def create_test_merge(l_test_parts):
    input_dir = get_output_dir()    
    assert input_dir.is_dir()

    output_dir = get_ready_data_dir()
    assert output_dir.is_dir()

    d_f = get_feature_converter()

    l_files = list (input_dir.iterdir())
    l_files = [x for x in l_files if x.suffix == '.npy']
   
    l_data_test = {}
    for zFeature in list (d_f.keys()):
        l_data_test[zFeature] = []


    l_iPart = []
    l_zVideo = []
    l_y = []

 
    for x in l_files:
        l_x = str(x.stem).split("_")

        isTestFile = (len (l_x) == 5) and (l_x[2] == 'Test')

        if isTestFile:
            pass
        else:
            continue

        iPart = int (l_x[1])
        video = l_x[3]
        y = l_x[4]

        isCollect = (iPart in l_test_parts)

        if isCollect:
            pass
        else:
            continue

        data = np.load(x)

        anFeature = data[:, 0]

        data = data[:, 1:]

        data = data.reshape(-1, 16, 3)

        num_rows = data.shape[0]
        assert num_rows % len (d_f.keys()) == 0

        num_rows_per_feature = num_rows // len (d_f.keys())

        l_iPart.extend([iPart] * num_rows_per_feature)
        l_zVideo.extend([video] * num_rows_per_feature)
        l_y.extend([y] * num_rows_per_feature)

        for zFeature in list (d_f.keys()):
            iF = d_f[zFeature]
            m_correct_feature = (anFeature == iF)
            l_data_test[zFeature].append(data[m_correct_feature])
            assert data[m_correct_feature].shape[0] == num_rows_per_feature

    num_meta = len (l_iPart)

    for zFeature in list (d_f.keys()):
        if len (l_data_test[zFeature]) > 0:
            anDataTest = np.concatenate(l_data_test[zFeature])
            assert anDataTest.shape[0] == num_meta
            np.save(output_dir / f"test_{zFeature}.npy", anDataTest)

    df_meta = pd.DataFrame({'iPart' : l_iPart, 'video': l_zVideo, 'y': l_y})

    df_meta.to_pickle(output_dir / f"test_meta.pkl")



def create_train_merge(l_train_parts):

    input_dir = get_output_dir()    
    assert input_dir.is_dir()

    output_dir = get_ready_data_dir()
    assert output_dir.is_dir()

    d_f = get_feature_converter()

    l_files = list (input_dir.iterdir())
    l_files = [x for x in l_files if x.suffix == '.npy']
   
    l_data_train = {}
    for zFeature in list (d_f.keys()):
        l_data_train[zFeature] = []

 
    for x in l_files:

        l_x = str(x.stem).split("_")

        isTrainFile = (len (l_x) == 5) and (l_x[2] == 'Train')
    
        if isTrainFile:
            pass
        else:
            continue

        iPart = int (l_x[1])
        original = l_x[3]
        fake = l_x[4]

        isCollect = (iPart in l_train_parts)

        if isCollect:
            pass
        else:
            continue

        data = np.load(x)

        anFeature = data[:, 0]

        data = data[:, 1:]

        data = data.reshape(-1, 32, 3)
    
        for zFeature in list (d_f.keys()):
            iF = d_f[zFeature]

            m_correct_feature = (anFeature == iF)

            l_data_train[zFeature].append(data[m_correct_feature])


    for zFeature in list (d_f.keys()):
        if len (l_data_train[zFeature]) > 0:
            anDataTrain = np.concatenate(l_data_train[zFeature])
            np.save(output_dir / f"train_{zFeature}.npy", anDataTrain)

            



