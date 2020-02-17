

import numpy as np
from mp4_frames import get_output_dir
from mp4_frames import get_ready_data_dir
from featureline import get_feature_converter
from featureline import is_error_line

import pandas as pd


####################################################################################
#
#   create_test_merge
#

def create_test_merge(iPartMin, iPartMax):

    assert iPartMax > iPartMin

    l_test_parts = list (range(iPartMin, iPartMax))

    num_length = 32

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

        isTestFile = (len (l_x) == 6) and (l_x[1] == 'Test')

        if isTestFile:
            pass
        else:
            continue

        iPart = int (l_x[3])
        video = l_x[4]
        y = l_x[5]

        isCollect = (iPart in l_test_parts)

        if isCollect:
            pass
        else:
            continue

        data = np.load(x)

        if is_error_line(data):
            continue

        anFeature = data[:, 0]

        data = data[:, 1:]

        data = data.reshape(-1, num_length, 3)

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
            np.save(output_dir / f"test_{zFeature}_p_{iPartMin}_p_{iPartMax}.npy", anDataTest)
        else:
            print(f"No data: test_{zFeature}_p_{iPartMin}_p_{iPartMax}")
            

    df_meta = pd.DataFrame({'iPart' : l_iPart, 'video': l_zVideo, 'y': l_y})

    df_meta.to_pickle(output_dir / f"test_meta_p_{iPartMin}_p_{iPartMax}.pkl")


####################################################################################
#
#   create_train_merge
#

def create_train_merge(iPartMin, iPartMax):

    assert iPartMax > iPartMin

    l_train_parts = list (range(iPartMin, iPartMax))


    num_length = 32


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

        isTrainFile = (len (l_x) == 6) and (l_x[1] == 'Pair')
    
        if isTrainFile:
            pass
        else:
            continue

        iPart = int (l_x[3])
        original = l_x[4]
        fake = l_x[5]

        isCollect = (iPart in l_train_parts)

        if isCollect:
            pass
        else:
            continue

        data = np.load(x)

        if is_error_line(data):
            continue

        anFeature = data[:, 0]

        data = data[:, 1:]

        data = data.reshape(-1, num_length * 2, 3) 
    
        for zFeature in list (d_f.keys()):
            iF = d_f[zFeature]

            m_correct_feature = (anFeature == iF)

            l_data_train[zFeature].append(data[m_correct_feature])


    for zFeature in list (d_f.keys()):
        if len (l_data_train[zFeature]) > 0:
            anDataTrain = np.concatenate(l_data_train[zFeature])
            np.save(output_dir / f"train_{zFeature}_p_{iPartMin}_p_{iPartMax}.npy", anDataTrain)

            
####################################################################################
#
#   create_train_merge_chunks
#

def create_train_merge_chunks(iPartMin, iPartMax):
    assert iPartMax > iPartMin
    l_Parts = list (range(iPartMin, iPartMax))

    for iPart in l_Parts:
        create_train_merge(iPart, iPart + 1)

####################################################################################
#
#   create_test_merge_chunks
#

def create_test_merge_chunks(iPartMin, iPartMax):
    assert iPartMax > iPartMin
    l_Parts = list (range(iPartMin, iPartMax))

    for iPart in l_Parts:
        create_test_merge(iPart, iPart + 1)





