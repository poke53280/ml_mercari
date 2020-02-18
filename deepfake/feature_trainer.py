

import numpy as np
from mp4_frames import get_output_dir
from mp4_frames import get_ready_data_dir
from featureline import get_feature_converter
from featureline import is_error_line

import pandas as pd
from random import shuffle

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



####################################################################################
#
#   create_train_chunks
#

def create_train_chunks(iPartMin, iPartMax, nGBInternal):
    assert iPartMax > iPartMin
    assert nGBInternal > 5

    data_dir = get_ready_data_dir()

    l_files = list (data_dir.iterdir())

    l_files_out = []

    for x in l_files:
        l_x = str(x.stem).split("_")
        if len(l_x) != 7:
            continue

        if l_x[0] != 'train':
            continue
            
        iMin = int (l_x[4])
        iMax = int (l_x[6])

        assert iMax > iMin

        if (iMin >= iPartMin) and (iMax <= iPartMax):
            pass
        else:
            continue

        l_files_out.append(x)


    shuffle(l_files_out)

    size_row_bytes = 64 * 3 * 4
    size_internal_bytes = nGBInternal * 1024 * 1024 * 1024

    max_internal_rows = int (size_internal_bytes / size_row_bytes)
    max_out_rows = 1000000

    l_data = []

    num_rows_internal = 0

    iFile = 0

    for idx, x in enumerate(l_files_out):
        isLastFile = (idx == (len(l_files_out) -1))

        print(f"loading {x}...")
        anData = np.load(x)

        assert anData.shape[0] <= max_internal_rows, "single file exceeds internal buffer size"

        num_rows_internal = num_rows_internal + anData.shape[0]
        l_data.append(anData.copy())

        if isLastFile or (num_rows_internal > max_internal_rows):
            print(f"Writing out. {num_rows_internal} > {max_internal_rows} or last file")
            anData = np.concatenate(l_data)
            np.random.shuffle(anData)

            num_rows_out = anData.shape[0]
        

            num_chunks = int (1 + num_rows_out / max_out_rows)

            print(f"   Writing out. {num_rows_out} lines in {num_chunks} chunks")

            l_data = np.array_split(anData, num_chunks)

            for data_chunk in l_data:
                file_out = data_dir / f"tr_{iPartMin}_{iPartMax}_{iFile:04}.npy"
                np.save(file_out, data_chunk)
                print(f" saved chunk with {data_chunk.shape[0]} lines")

                iFile = iFile + 1

            l_data = []
            num_rows_internal = 0


####################################################################################
#
#   _get_meta_file
#

def _get_meta_file(iMin, iMax):
    data_dir = get_ready_data_dir()
    filename = data_dir / f"test_meta_p_{iMin}_p_{iMax}.pkl"
    return filename

        
####################################################################################
#
#   create_test_video_chunks
#

def create_test_video_chunks(iPartMin, iPartMax):

    assert iPartMax > iPartMin

    data_dir = get_ready_data_dir()

    l_files = list (data_dir.iterdir())

    l_files_out = []

    for x in l_files:

        l_x = str(x.stem).split("_")
        if len(l_x) != 7:
            continue

        if l_x[0] != 'test':
            continue

        if l_x[1] == 'meta':
            continue
            
        iMin = int (l_x[4])
        iMax = int (l_x[6])

        assert iMax > iMin

        if (iMin >= iPartMin) and (iMax <= iPartMax):
            pass
        else:
            continue

        metafile = _get_meta_file(iMin, iMax)
        
        if metafile.is_file():
            pass
        else:
            continue

        l_files_out.append((x, metafile))

    """c"""

    l_test = []
    l_meta = []
    
    for x in l_files_out:
        anTest = np.load(x[0])
        df_meta = pd.read_pickle(x[1])

        assert anTest.shape[0] == df_meta.shape[0]

        l_test.append(anTest)
        l_meta.append(df_meta)

    anTest = np.concatenate(l_test)
    df_meta = pd.concat(l_meta, ignore_index = True)

    z_video = df_meta.iPart.astype('str') + "_" + df_meta.video

    azVideo = np.unique(z_video)

    for ix, x in enumerate(azVideo):
        m = z_video == x
        anVideoData = anTest[m]
        zRealFake = df_meta[m].y.iloc[0]
        zOut = data_dir / f"te_{iPartMin}_{iPartMax}_{ix:04}_{zRealFake}"

        np.save(zOut, anVideoData)


