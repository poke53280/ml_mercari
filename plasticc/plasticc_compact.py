
DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

import gc
import numpy as np
import pandas as pd


data = pd.read_csv(DATA_DIR + 'test_set.csv')


data = pd.read_csv(DATA_DIR + 'training_set.csv')


# Remove detected, currently not in use

data = data.drop(['detected'], axis = 1)
gc.collect()

data = data.assign(object_id = pd.to_numeric(data.object_id, downcast = 'signed'))
gc.collect()


data = data.assign(passband = pd.to_numeric(data.passband, downcast = 'unsigned'))
gc.collect()


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
