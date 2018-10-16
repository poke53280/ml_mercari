
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

local_dir = os.getenv('LOCAL_PY_DIR')

assert local_dir is not None, "Set environment variable LOCAL_PY_DIR to parent folder of ai_lab_datapipe folder. Instructions in code."

# * 'Windows/Start'button
# * Type 'environment...'. 
# * Select option 'Edit environment variables for your account' (NOT: 'Edit the system environment variables)
# * New. LOCAL_PY_DIR. Value is parent folder to ai_lab_datapipe
# * Restart IDE/python context. 

print(f"Local python top directoy is set to {local_dir}")
os.chdir(local_dir)


from ai_lab_datapipe.sf_pipeline.TimeLineTool import *


def get_flux_sample(x_mjd, y_flux, y_std):
    assert x_mjd.shape[0] == y_flux.shape[0] == y_std.shape[0], "inequal array lengths"

    y_out = np.empty(shape=y_flux.shape[0], dtype = float)

    for idx in range (x_mjd.shape[0]):

        flux = np.random.normal(y_flux[idx], y_std[idx])
        y_out[idx] = flux

    return y_out



DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)



training = pd.read_csv(DATA_DIR + 'training_set.csv')

meta_training = pd.read_csv(DATA_DIR + 'training_set_metadata.csv')



m = training.passband == 0


training = training[m]


m = training.object_id == 1920

q = training[m]

m = q.passband == 0

q = q[m]




df_res = pd.DataFrame({'object_id' :all_IDs})   



w2 = df_res.apply(get_sequence, axis = 1, args = (training, 3, 1000))

# Return generation data as well.

# Likely bug: Check out '1000' setting above - bad numbers.



def get_sequence(x, df, internal_width, num_samples):

    nStepSize = internal_width/num_samples

    m = df.object_id == x['object_id']

    q = df[m]

    sorted = get_groups(q, internal_width)
    
    y = np.bincount(sorted)

    idx_max = np.argmax(y)

    m = sorted == idx_max

    
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

        for i in range(nSamples):
            x_sample = min_sample + i * nStepSize

            value = f(x_sample)
            output[i] = value


    else:
       center_min = x_mjd.min() + 0.5 * internal_width
       center_max = x_mjd.max() - 0.5 * internal_width

       center = np.random.uniform(low = center_min, high = center_max)

       min_sample = center - internal_width/2.0
       max_sample = center + internal_width/2.0

       for i in range(nSamples):
            x_sample = min_sample + i * nStepSize

            x_sample = min(x_sample, x_mjd.max())
            x_sample = max(x_sample, x_mjd.min())

            value = f(x_sample)
            output[i] = value


    output_idx = range(output.shape[0])

    d = dict(zip(output_idx, output))

    return pd.Series(d)

"""c"""





    






for i in range(nSamples):





testing = pd.read_csv(DATA_DIR + 'test_set.csv')

meta_testing = pd.read_csv(DATA_DIR + 'test_set_metadata.csv')

m = testing.object_id < 448280
testing = testing[m]


un_testing = np.unique(testing.object_id)

un_testing.shape




training.sample(5)

meta_training.sample(5)


#ddf
m = meta_training.ddf == 1

q = meta_training[m]

len (np.unique(meta_training.object_id))

l = []

for id in un_testing:


    m = testing.object_id == id

    q = testing[m]

    q = q.sort_values(by = ['passband', 'mjd'])

    m = q.passband == 0

    q = q[m]

    res = get_num_large_groups(q, 3, 40)

    l.append(res)

    print(f"res = {res}")



an = np.array(l)

np.min(an)
np.max(an)



analyze_groups(q, 3, 40)


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



    




    






