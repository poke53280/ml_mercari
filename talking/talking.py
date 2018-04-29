
#
# Load and preprocess talking
#
#


import pandas as pd
import numpy as np
import datetime
import pylab as pl
import gc

import general.TimeLineTool as tl
import general.TimeAndDate as td

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\talking-data\\"            
DATA_DIR_BASEMENT = "c:\\data_talking\\"
DATA_DIR = DATA_DIR_PORTABLE


# Read pickle
df = pd.read_pickle(DATA_DIR + "preprocessed.pkl")



# Create pickle

isLoadTestSample = False



# XXX Configure column sizes on load

if isLoadTestSample:
    print('loading test sample data...')
    df = pd.read_csv(DATA_DIR + "train_sample.csv")

else:
    print('loading train data...')
    df = pd.read_csv(DATA_DIR + "train.csv")


df = df.drop(['is_attributed'], axis = 1)

df['time'] = td.TimeAndDate_GetSecondsSinceEpochSeries(df.click_time)

MIN_EPOCH = df.time.min()
df['time'] = df.time - MIN_EPOCH
df['time'] = pd.to_numeric(df.time, downcast = 'integer')

df = df.drop(['click_time'], axis = 1)

df.attributed_time = df.attributed_time.fillna(0)
df.attributed_time = td.TimeAndDate_GetSecondsSinceEpochSeries(df.attributed_time)
df.attributed_time = df.attributed_time - MIN_EPOCH
df.attributed_time = df.attributed_time.replace(- MIN_EPOCH, 0)
df['attributed_time'] = pd.to_numeric(df.attributed_time, downcast = 'integer')

print('loading test supplement data...')

df_s = pd.read_csv(DATA_DIR + "test_supplement.csv")

df_s = df_s.drop(['click_id'], axis = 1)

df_s['time'] = td.TimeAndDate_GetSecondsSinceEpochSeries(df_s.click_time)

df_s['time'] = df_s.time - MIN_EPOCH
df_s['time'] = pd.to_numeric(df_s.time, downcast = 'integer')

df_s = df_s.drop(['click_time'], axis = 1)

df_s = df_s.reset_index()

numpy_type = df.attributed_time.dtype

acTestMarker = np.zeros(len (df_s), dtype = numpy_type) -1

df_s = df_s.assign(attributed_time=acTestMarker)
df_s = df_s.drop(['index'], axis = 1)

df_s = df_s[ ['ip', 'app', 'device', 'os', 'channel', 'attributed_time', 'time']]


df = pd.concat([df, df_s], ignore_index=True)

del df_s
gc.collect(0)



nCut = 0

if nCut > 0:
    df = df[:nCut]
    # Cut fully away largest IP (possibly partly cut now)
    m = (df.ip == df.ip.max())
    df = df[~m]

gc.collect(0)

# 'system' : [device, os]
df['temp'] = df.device * df.os.max() + df.os
df['temp'] = pd.Categorical(df.temp)
df['system'] = df.temp.cat.codes

df = df.drop(['temp'], axis = 1)

# Drop device and os, at least for now
df = df.drop(['device', 'os'], axis = 1)


# [ip, system]
df['temp'] = df.ip * df.system.max() + df.system
df['temp'] = pd.Categorical(df.temp)
df['ip_and_sys'] = df.temp.cat.codes
df = df.drop(['temp'], axis = 1)

# [system, channel]

df['temp'] = df.channel * df.system.max() + df.system
df['temp'] = pd.Categorical(df.temp)
df['channel_and_sys'] = df.temp.cat.codes
df = df.drop(['temp'], axis = 1)

# [app, system, channel]
df['temp'] = df.app * df.channel_and_sys.max() + df.channel_and_sys
df['temp'] = pd.Categorical(df.temp)
df['app_channel_sys'] = df.temp.cat.codes
df = df.drop(['temp'], axis = 1)


# [ip, system, channel]
df['temp'] = df.ip * df.channel_and_sys.max() + df.channel_and_sys
df['temp'] = pd.Categorical(df.temp)
df['ip_sys_channel'] = df.temp.cat.codes
df = df.drop(['temp'], axis = 1)


# [ip, app, system, channel]
df['temp'] = df.ip * df.app_channel_sys.max() + df.app_channel_sys
df['temp'] = pd.Categorical(df.temp)
df['ip_app_sys_channel'] = df.temp.cat.codes
df = df.drop(['temp'], axis = 1)

# Drop all elementals for now.
df = df.drop(['ip', 'app', 'channel', 'system'], axis = 1)

gc.collect(0)

df = df.sort_values(by = ['time'])


df.to_pickle(DATA_DIR + "preprocessed.pkl")



