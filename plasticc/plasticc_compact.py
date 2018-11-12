
DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_SERVER = "D:\\t149900\\"
DATA_DIR = DATA_DIR_SERVER

import gc
import numpy as np
import pandas as pd


def get_cluster_info(x, proximity):

    xd = np.diff(x)
    m = xd < proximity
    aw = np.where(~m)[0]

    g_high_idx_inclusive = np.append(aw, x.shape[0] -1)
    low_idx_inclusive = np.append(0, aw + 1)

    num_elements = g_high_idx_inclusive - low_idx_inclusive + 1 

    x_min = x[low_idx_inclusive]
    x_max = x[g_high_idx_inclusive]

    x_diff = x_max - x_min

    return x_diff, num_elements
"""c"""

def cluster_info(id, closeness):

    m_data = (data.object_id == id)

    x = data[m_data].mjd.values

    l, n = get_cluster_info(x, closeness)

    assert l.shape == n.shape
    assert n.sum() == x.shape[0]

    idx = np.argsort(n)[::-1]

    n = n[idx]
    l = l[idx]

    n = np.append(n, np.zeros(4))
    l = np.append(l, np.zeros(4))
    n = n[:4]
    l = l[:4]

    return l, n, x.shape[0]

"""c"""



pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

meta = pd.read_csv(DATA_DIR + 'training_set_metadata.csv')
data = pd.read_csv(DATA_DIR + 'training_set.csv')

m = meta.ddf == 0

meta = meta[m]


ids = meta.object_id.values

al = np.empty((ids.shape[0], 4), dtype = np.float32)
an = np.empty((ids.shape[0], 4), dtype = np.int32)

samples = np.empty(ids.shape[0], dtype = np.int32)

closeness = 10

for idx, id in enumerate(ids):
    id = np.random.choice(ids)
    l, n, s = cluster_info(id, closeness)

    al[idx, :] = l[:4]
    an[idx, :] = n[:4]
    samples[idx] = s

meta = meta.assign(n = samples)

meta = meta.assign(n_0 = an[:, 0])
meta = meta.assign(l_0 = al[:, 0])

meta = meta.assign(n_1 = an[:, 1])
meta = meta.assign(l_1 = al[:, 1])

meta = meta.assign(n_2 = an[:, 2])
meta = meta.assign(l_2 = al[:, 2])

meta = meta.assign(n_3 = an[:, 3])
meta = meta.assign(l_3 = al[:, 3])


meta.n_0.describe()
meta.l_0.describe()

meta.n_1.describe()
meta.l_1.describe()

meta.n_2.describe()
meta.l_2.describe()



# train: data.mjd.min()
# 59580.0343

# test: data.mjd.min()
# 59580.0338   <= The Smallest. USE FOR TEST AND TRAIN


MIN_MDJ = 59580.0338


data.mjd -= MIN_MDJ

print(f"mjd min = {data.mjd.min()}, mjd max = {data.mjd.max()}")

data.mjd *= 10000

print(f"mjd min = {data.mjd.min()}, mjd max = {data.mjd.max()}")

#                          10943292
# mjd min = 0.0, mjd max = 10943286.999999981
#   10, 943, 286
# 2^32 = 4,294,967,296

umjd = data.mjd.astype(np.int32)

print(f"mjd min = {umjd.min()}, mjd max = {umjd.max()}")

data = data.assign(umjd = umjd)

data = data.drop(['mjd'], axis = 1)
gc.collect()

data.flux.min()
data.flux.max()

m = data.flux < -5000

m.value_counts()

m = data.flux > 10000

m.value_counts()

flux = data.flux.values

flux = np.clip(flux, -5000, 10000)

flux += 5000

max_flux = 15000

rScaleDown = max_flux/ 65535

print(f"Flux scale down = {rScaleDown}")

flux /= rScaleDown


nFlux = flux.astype(np.uint16)

nFlux.min()
nFlux.max()

data = data.assign(uFlux = nFlux)

data = data.drop(['flux'], axis = 1)
gc.collect()

data.flux_err /= rScaleDown

m = data.flux_err > 500

m.value_counts()

flux_err = np.clip(data.flux_err, 0, 500)

# Remember this one for scale up
flux_err_scale_down = 500/ 255.0

flux_err /= flux_err_scale_down

flux_err.min()
flux_err.max()

nFluxErr = flux_err.astype(np.uint8)

data = data.assign(uFluxErr = nFluxErr)

data = data.drop(['flux_err'], axis = 1)
gc.collect()

data.to_pickle(DATA_DIR + "compact_training_data.pkl")
data.to_pickle(DATA_DIR + "compact_test_data.pkl")


# Remember this one for scale up
#flux_err_scale_down = 500/ 255.0
#
#flux_err /= flux_err_scale_down
