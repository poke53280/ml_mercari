

import pandas as pd
import numpy as np
import datetime
import pylab as pl
import gc

import general.TimeLineTool as tl
import general.TimeAndDate as td


isLoadTestSample = False

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\talking-data\\"            
DATA_DIR_BASEMENT = "c:\\data_talking\\"
DATA_DIR = DATA_DIR_PORTABLE

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

#df.isnull().sum()
# No nulls

# Popular ips:

#
#    5348      1742881
# 5314      1619348
# 5147       252546
# 114220     210428

#m = df.ip == 114220

#df1 = df[m]

#acData = df1.time.values

# goto TimeLineTool.py with the 114220 data





#q = df[m1 & m2]
#len (q)

#q = q.sort_values(['ip'])

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

#
#
# Cluster analysis in TimeLineTool.py
#
#

d = tl.TimeLineTool_analyse_user_code(df, 2182)

n_clusters = d.keys()
n_gap = d.values()


pl.plot(n_gap,n_clusters)
pl.xlabel('Gap slack')
pl.ylabel('Clusters')
pl.show()

idx = 92
m = df.user_code == idx

q = df[m]

len (q)

s = q.time

acTime = np.array(s)

acTime = acTime - acTime.min()

min = acTime.min()
max = acTime.max()

interval_length = max - min


pl.hist(acShort, bins = bin_size)



pl.show()

diff = np.diff(s)

m = diff < 5 * 60

diff = diff[m] 

pl.hist(diff, bins = 5 * 60)

pl.show()


#!!! Compare with mixed user codes to see if there is a signal



df = df.assign(session=res)

df = df.drop(['user_code'], axis = 1)


q = df [:90]
groups = df.groupby(q.session)
groups.groups


pd.pivot_table(q,index=["session"])...

