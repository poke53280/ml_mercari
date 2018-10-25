
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import sys
import itertools

from datetime import datetime



def get_flux_sample(x_mjd, y_flux, y_std):
    assert x_mjd.shape[0] == y_flux.shape[0] == y_std.shape[0], "inequal array lengths"

    y_out = np.empty(shape=y_flux.shape[0], dtype = float)

    for idx in range (x_mjd.shape[0]):

        flux = np.random.normal(y_flux[idx], y_std[idx])
        y_out[idx] = flux

    return y_out



def get_sequence_float32_sequence(q, num_iters, l_bandwidth, num_samples):

    ar_mjd_all = np.array(q.umjd, dtype = np.float32)

    ar_mjd_all /= 1000.0

    mjd_min = np.min(ar_mjd_all)
    mjd_max = np.max(ar_mjd_all)

    mjd_length = mjd_max - mjd_min

    func_x = []
    func_y = []

    flux_err_scale_down = 500/ 255.0

    for iPassband in range(6):

        q_b = q[q.passband == iPassband]

        ar_mjd = np.array(q_b.umjd, dtype = np.float32)

        ar_mjd /= 1000.0

        ar_flux = np.array(q_b.uFlux, dtype = np.float32)
        ar_fluxerr = np.array(q_b.uFluxErr, dtype = np.float32)

        ar_fluxerr *= flux_err_scale_down

        y_out = get_flux_sample(ar_mjd, ar_flux, ar_fluxerr)

        func_x.append(ar_mjd)
        func_y.append(y_out)

    """c"""
    
    afRes = np.empty(shape = (num_iters, num_samples * len (l_bandwidth) * 6), dtype = np.float32)

    for idx in range (num_iters):

        afResOffset = 0

        for b in l_bandwidth:

            if b > mjd_length:
                print(f"low = {mjd_min - b + mjd_max - mjd_min}, high = {mjd_min}")
                mjd_start = np.random.uniform(low = mjd_min - b + mjd_length, high = mjd_min)
            else:
                mjd_start = np.random.uniform(low = mjd_min, high = mjd_max - b)

            x_samples = np.linspace(mjd_start, mjd_start + b, num_samples)

            for iPassband in range(6):

                fSampleValues = np.interp(x_samples, func_x[iPassband], func_y[iPassband], left = y_out[0], right = y_out[-1])

                isUINT8 = False
                if isUINT8:
                    fSampleValues -= fSampleValues.min()

                    if fSampleValues.max() > 0:
                        fSampleValues  /= fSampleValues.max()

                    fSampleValues *= 255.0

                    fSampleValues = fSampleValues.astype(np.uint8)

                afRes[idx, afResOffset:afResOffset + num_samples] = fSampleValues
                afResOffset += num_samples

        """c"""

    """c"""
    return afRes


def generate_dataset(meta, data, num_iters, l_bandwidth, num_samples):

    start = datetime.now()

    nItems = meta.shape[0]

    print(f"Generating {num_iters} x {nItems} (iters x items) rows of item data...")
    
    isTrain = 'target' in meta.columns

    ids = np.array(meta.object_id, dtype=np.int32)

    res = generate_dataset_array(ids, data, num_iters, l_bandwidth, num_samples)

    print("---0---")
    
    df_res = pd.DataFrame(data = res)

    print("---1---")

    ids_repeated = np.repeat(ids, num_iters)

    print("---2---")

    df = pd.DataFrame({'object_id' :ids_repeated})

    print("---3---")

    if isTrain:
        meta_id_target = meta[['object_id', 'target']]
        df = df.merge(meta_id_target, how = 'left', left_on= 'object_id', right_on = 'object_id')

    print("---4---")

  
    df_res = df_res.assign(object_id = df['object_id'])

    if isTrain:
        df_res = df_res.assign(target = df['target'])
   

    print("---5---")

    print("Dataset generation done.")

    end = datetime.now()

    dT = end - start
    
    dSeconds = dT.total_seconds()

    nRowsPerSecond = (num_iters * nItems)/ dSeconds
    nItemsPerSecond = nItems / dSeconds

    total_item_count = 3492890

    rPredictedTimeSecs = dSeconds * total_item_count / nItems

    rPredictedTimeHrs = rPredictedTimeSecs/3600.0

    print(f"Time spent generating {num_iters} x {nItems}: {dSeconds:.1f}s.  rows/s: {nRowsPerSecond:.1f} items/s: {nItemsPerSecond:.1f}")

    print(f"   => Prediction for all {total_item_count} items: {rPredictedTimeHrs:.1f}h")

    
    return df_res

"""c"""

def generate_dataset_array(ids, data, num_iters, l_bandwidth, num_samples):

    element_size = num_samples * len(l_bandwidth) * 6

    nItems = ids.shape[0]

    afRes = np.empty(shape=(nItems * num_iters, element_size), dtype = np.float32)

    afResOffset = 0

    for idx, id in enumerate(ids):
   
        if idx % 100 == 0:
            print(f"Processing item {idx}/{nItems}...")

        m = data.object_id == id

        #q = data[m].copy()

        res = get_sequence_float32_sequence(data[m], num_iters, l_bandwidth, num_samples)

        afRes[num_iters*idx:num_iters*idx+num_iters, :] = res

    return afRes



DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

l_bandwidth = [1, 2, 3, 5, 10, 30, 50, 100, 200, 400]
num_samples = 100

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)




def generate_test_slice(iSplit, desc):
    
    data_filename = DATA_DIR + desc + "_test_" + str(iSplit) + ".pkl"
    meta_filename = DATA_DIR + desc + "_test_meta_" + str(iSplit) + ".pkl"

    data = pd.read_pickle(data_filename)
    meta = pd.read_pickle(meta_filename)

    num_iters = 10

    df = generate_dataset(meta, data, num_iters, l_bandwidth, num_samples)

    filename = DATA_DIR + "test_processed" + desc + "_" + str(iSplit)+ ".pkl"

    df.to_pickle(filename)

    print(f"Saved dataset as '{filename}'")


def generate_slice(iSplit, filename_desc, target_class):

    num_items_each = 200000

    training_filename = DATA_DIR + filename_desc + "_training_" + str(iSplit) + ".pkl"
    
    training = pd.read_pickle(training_filename)

    meta_training = pd.read_pickle(DATA_DIR + filename_desc + "_training_meta_" + str(iSplit) + ".pkl")

    m_true = (meta_training.target == target_class)
   
    nTrue = meta_training[m_true].shape[0]
    nFalse = meta_training[~m_true].shape[0]

    num_true_iters = int(0.5 + num_items_each / nTrue)
    num_false_iters = int(0.5 + num_items_each / nFalse)

    df_true = generate_dataset(meta_training[m_true], training, num_true_iters, l_bandwidth, num_samples)
    df_false = generate_dataset(meta_training[~m_true], training, num_false_iters, l_bandwidth, num_samples)

    print("Concating true and false datasets...")

    df = pd.concat([df_true, df_false], axis = 0)

    print("Concating done.")      

    anTarget = np.array(df.target, dtype = int)

    print("2 done.")  

    m = anTarget == target_class

    print("3 done.")

    anTarget90 = np.empty(shape = anTarget.shape, dtype = int)

    print("4 done.")

    anTarget90[m] = 1

    print("5 done.")

    anTarget90[~m] = 0

    print("6 done.")

    df = df.assign(target_90 = anTarget90)

    print("7 done.")

    df = df.sample(frac=1).reset_index(drop=True)

    print("8 doneti.")

    df = df.drop(['target'], axis = 1)

    print("9 done.")

    anIDs = np.unique(df.object_id)

    print(f"#Unique objects in training set: {anIDs.shape[0]}")

    filename = DATA_DIR + "df_t_" + filename_desc + "_" + str(iSplit)+ ".pkl"

    print(f"Saving dataset as '{filename}'...")

    df.to_pickle(filename)

    print(f"Saved dataset done.")


def main():
    
    l = []
    for arg in sys.argv[1:]:
        l.append(arg)

    

    trte = l[0]

    assert (trte == "te") or (trte == "tr")

    isTest = (trte == "te")

    desc = l[1]
    i_split = int(l[2])
    i_targetclass = int(l[3])

    print(f"test: {isTest}")

    print(f"desc: {desc}")
    print(f"split idx {i_split}")
    print(f"target class: {i_targetclass}")

    assert i_split >= 0

    if isTest:
        generate_test_slice(i_split, desc)
    else:
        generate_slice(i_split, desc, i_targetclass)
 


if __name__ == "__main__":
    main()

# TODO: MOVE TRANSFORM OUT TO CALC FORMAT IN COMPACT

def dev_start():
    meta = pd.read_csv(DATA_DIR + 'training_set_metadata.csv')
    data = pd.read_csv(DATA_DIR + 'training_set.csv')

    MIN_MDJ = 59580.0338
    rFluxScaleDown = 15000.0/ 65535
    flux_err_scale_down = 500/ 255.0
    
   
    data_c = pd.read_pickle(DATA_DIR + 'compact_training_data.pkl')

    mjd = ((data_c.umjd / 10000.0) + MIN_MDJ).astype(np.float32)
    
    flux = ((data_c.uFlux * rFluxScaleDown) - 5000).astype(np.float32)

    flux_err = (data_c.uFluxErr * flux_err_scale_down * rFluxScaleDown).astype(np.float32)

    # CONTINUE HERE


    id = 248547

    m = (data.object_id == id)

    q = data[m]

    q_c = data_c[m].copy()

    q_c.dtypes
  


    
