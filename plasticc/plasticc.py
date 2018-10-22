
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import sys
import itertools



def get_flux_sample(x_mjd, y_flux, y_std):
    assert x_mjd.shape[0] == y_flux.shape[0] == y_std.shape[0], "inequal array lengths"

    y_out = np.empty(shape=y_flux.shape[0], dtype = float)

    for idx in range (x_mjd.shape[0]):

        flux = np.random.normal(y_flux[idx], y_std[idx])
        y_out[idx] = flux

    return y_out



def get_sequence_fast(x, df):

    l_bandwidth = [3, 30, 50]

    m = training.object_id == 615

    q = training[m].copy()

    p = []

    for iPassband in range(6):

        d = {}

        m_b = q.passband == iPassband
        q_b = q[m_b].copy()

        ar_mjd = np.array(q_b.mjd)
        ar_flux = np.array(q_b.flux)
        ar_fluxerr = np.array(q_b.flux_err)

        d["mjd"] = ar_mjd
        d["flux"] = ar_flux
        d["flux_err"] = ar_fluxerr
        
        d["groups"] = []

        nd = np.diff(ar_mjd)
        
        for b in l_bandwidth:
            
            m = nd > b
            sorted = np.empty(shape = len(ar_mjd), dtype = np.int)
            sorted[1:] = m.cumsum()
            sorted[0] = 0

            d["groups"].append(sorted)

        p.append(d)
        CONTINUE HERE...
    
    l_num_samples = [10, 100]
    
    
    #list(itertools.product(*[l_passbands, l_num_samples, l_internal_width]))

   

    



def get_sequence(x, df, internal_width, num_samples, n_passband, col_prefix):

    nStepSize = internal_width/num_samples

    m = (df.object_id == x['object_id']) & (df.passband == n_passband)

    q = df[m]

    num_points = len (q)

    sorted = get_groups(q, internal_width)
    
    y = np.bincount(sorted)

    idx_max = np.argmax(y)

    m = sorted == idx_max
    
    num_max_points = m.sum()

    num_groups = len(y)

    used_group_idx = np.random.choice(range(num_groups))

    m = sorted == used_group_idx

    num_points_used_group = m.sum()
    
    x_mjd = np.array(q[m].mjd)
    y_flux = np.array(q[m].flux)
    y_std = np.array(q[m].flux_err)

    y_out = get_flux_sample(x_mjd, y_flux, y_std)

    L_full_width = x_mjd.max() - x_mjd.min()

    output = np.empty(shape = num_samples, dtype = np.float)

    if m.sum() == 1:
        f = lambda x: y_out[0]
    else:
        f = interp1d(x_mjd, y_out)

    if L_full_width >= internal_width:

        center_min = x_mjd.min() + 0.5 * internal_width
        center_max = x_mjd.max() - 0.5 * internal_width

        center = np.random.uniform(low = center_min, high = center_max)

        min_sample = center - internal_width/2.0

        assert min_sample >= x_mjd.min(), f"{min_sample} >= {x_mjd.min()}, id = {object_id}"

        max_sample = center + internal_width/2.0

        for i in range(num_samples):
            x_sample = min_sample + i * nStepSize

            value = f(x_sample)
            output[i] = value

    else:
       center_min = x_mjd.min() + 0.5 * internal_width
       center_max = x_mjd.max() - 0.5 * internal_width

       center = np.random.uniform(low = center_min, high = center_max)

       min_sample = center - internal_width/2.0
       max_sample = center + internal_width/2.0

       for i in range(num_samples):
            x_sample = min_sample + i * nStepSize

            x_sample = min(x_sample, x_mjd.max())
            x_sample = max(x_sample, x_mjd.min())

            value = f(x_sample)
            output[i] = value

    d = {}

    output_idx = range(output.shape[0])


    d['flux_min'] = np.min(output)
    d['flux_max'] = np.max(output)
    d['flux_mean'] = np.mean(output)
    d['flux_var'] = np.var(output)

    d['num_points_group_max'] = num_max_points

    d['t_width'] = internal_width
    d['t_min'] = min_sample

    d['num_points'] = num_points

    d['num_groups'] = num_groups

    d['group_used_idx'] = used_group_idx

    d['num_points_used_group'] = num_points_used_group

    # output_gauss = rank_gauss(output)

    s = MinMaxScaler()

    output_scaled = s.fit_transform(output.reshape(-1, 1)).squeeze()

    d.update(dict(zip(output_idx, output_scaled)))


    k = d.keys()
    v = d.values()

    k = [col_prefix + str(x) for x in k]

    d = dict(zip (k, v))


    return pd.Series(d)

"""c"""


def get_groups(q, bandwidth):
     a = np.array(q.mjd)
     nd = np.diff(a)
     m = nd > bandwidth
     sorted = np.empty(shape = len(a), dtype = np.int)
     sorted[1:] = m.cumsum()
     sorted[0] = 0

     return sorted


def get_num_large_groups(q, group_threshold, bandwidth):
    sorted = get_groups(q, bandwidth)
    y = np.bincount(sorted)
    m = y >= group_threshold
    return m.sum()



def generate_dataset_fast(meta_filtered, training, num_samples):
    
    ids = np.array(meta_filtered.object_id)

    id_sampled = np.random.choice(ids, size = num_samples, replace=True)

    df = pd.DataFrame({'object_id' :id_sampled})

    df_slim = df.copy()

    df_sample = df_slim.apply(get_sequence_fast, axis = 1, args = (training))

    df = pd.concat([df, df_sample], axis = 1)

    meta_id_target = meta_filtered[['object_id', 'target']]
   
    df = df.merge(meta_id_target, how = 'left', left_on= 'object_id', right_on = 'object_id')
    
    return df







def generate_dataset(meta_filtered, training, num_samples):

    ids = np.array(meta_filtered.object_id)

    id_sampled = np.random.choice(ids, size = num_samples, replace=True)

    df = pd.DataFrame({'object_id' :id_sampled})

    df_slim = df.copy()

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 3, 100, 0, '0'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 3, 100, 1, '1'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 3, 100, 2, '2'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 3, 100, 3, '3'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 3, 100, 4, '4'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 3, 100, 5, '5'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 30, 100, 0, '6'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 30, 100, 1, '7'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 30, 100, 2, '8'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 30, 100, 3, '9'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 30, 100, 4, '10'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 30, 100, 5, '11'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 50, 100, 0, '12'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 50, 100, 1, '13'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 50, 100, 2, '14'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 50, 100, 3, '15'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 50, 100, 4, '16'))
    df = pd.concat([df, df_sample], axis = 1)

    df_sample = df_slim.apply(get_sequence, axis = 1, args = (training, 50, 100, 5, '17'))
    df = pd.concat([df, df_sample], axis = 1)

    meta_id_target = meta_filtered[['object_id', 'target']]
   
    df = df.merge(meta_id_target, how = 'left', left_on= 'object_id', right_on = 'object_id')
    
    return df


def analyze_groups(q, group_threshold, bandwidth):

    plt.plot(q.mjd, q.flux)

    a = np.array(q.mjd)

    nd = np.diff(a)

    m = nd > bandwidth

    sorted = np.empty(shape = len(a), dtype = np.int)

    sorted[1:] = m.cumsum()
    sorted[0] = 0

    y = np.bincount(sorted)

    m = y >= group_threshold

    print(f"#Groups >= 3: {m.sum()}")

    for idx in range(len(sorted)):

        m = sorted == idx

        idx = np.where(m)[0]

        if idx.shape[0] < 2:
            continue

        print(f"idx = {idx}")
     
        x_mjd = np.array(q[m].mjd)
        y_flux = np.array(q[m].flux)
        y_std = np.array(q[m].flux_err)

        y_out = get_flux_sample(x_mjd, y_flux, y_std)

        f = interp1d(x_mjd, y_out)

        xnew = np.linspace(x_mjd.min(), x_mjd.max(), num=41, endpoint=True)

        plt.plot(x_mjd, y_out, 'o', xnew, f(xnew), '-')
        plt.legend(['data', 'linear'], loc='best')

    plt.show()



DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

def generate_slice(iSplit, filename_desc, target_class):
 

    training_filename = DATA_DIR + filename_desc + "_training_" + str(iSplit) + ".pkl"

    training = pd.read_pickle(training_filename)

    meta_training = pd.read_pickle(DATA_DIR + filename_desc + "_meta_" + str(iSplit) + ".pkl")

    IDs = meta_training.object_id

    assert IDs.shape == np.unique(IDs).shape

    m_true = (meta_training.target == target_class)


    df_true = generate_dataset(meta_training[m_true], training, 20000)
    df_false = generate_dataset(meta_training[~m_true], training, 20000)

    df = pd.concat([df_true, df_false], axis = 0)

    anTarget = np.array(df.target, dtype = int)

    m = anTarget == target_class

    anTarget90 = np.empty(shape = anTarget.shape, dtype = int)

    anTarget90[m] = 1
    anTarget90[~m] = 0

    df = df.assign(target_90 = anTarget90)

    df = df.sample(frac=1).reset_index(drop=True)

    df = df.drop(['target'], axis = 1)


    anIDs = np.unique(df.object_id)

    print(f"#Unique objects in training set: {anIDs.shape[0]}")


    filename = DATA_DIR + "df_t_" + filename_desc + "_" + str(iSplit)+ ".pkl"

    df.to_pickle(filename)

    print(f"Saved dataset as '{filename}'")




def main():
    
    l = []
    for arg in sys.argv[1:]:
        l.append(arg)

    desc = l[0]
    i_split = int(l[1])
    i_targetclass = int(l[2])

    print(f"desc: {desc}")
    print(f"split idx {i_split}")
    print(f"target class: {i_targetclass}")

    assert i_split >= 0

    generate_slice(i_split, desc, i_targetclass)
 




if __name__ == "__main__":
    main()