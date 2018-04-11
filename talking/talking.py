

import pandas as pd
import numpy as np
import datetime
import pylab as pl
import gc

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\talking-data\\"            
DATA_DIR_BASEMENT = "c:\\data_talking\\"
DATA_DIR = DATA_DIR_PORTABLE


print('loading train data...')

df = pd.read_csv(DATA_DIR + "train.csv")
df = df.drop(['is_attributed'], axis = 1)



df['time'] = TimeAndDate_GetSecondsSinceEpochSeries(df.click_time)

MIN_EPOCH = df.time.min()
df['time'] = df.time - MIN_EPOCH
df['time'] = pd.to_numeric(df.time, downcast = 'integer')

df = df.drop(['click_time'], axis = 1)

df.attributed_time = df.attributed_time.fillna(0)
df.attributed_time = TimeAndDate_GetSecondsSinceEpochSeries(df.attributed_time)
df.attributed_time = df.attributed_time - MIN_EPOCH
df.attributed_time = df.attributed_time.replace(- MIN_EPOCH, 0)
df['attributed_time'] = pd.to_numeric(df.attributed_time, downcast = 'integer')


print('loading test data...')

df_s = pd.read_csv(DATA_DIR + "test_supplement.csv")

df_s = df_s.drop(['click_id'], axis = 1)

df_s['time'] = TimeAndDate_GetSecondsSinceEpochSeries(df_s.click_time)

df_s['time'] = df_s.time - MIN_EPOCH
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


print('sorting combined data...')

df = df.sort_values(by = ['time'])


m1 = df.time > 59900
m2 = df.time < 59910

q = df[m1 & m2]

len (q)

q = q.sort_values(['ip', 'os'])








qif nCut > 0:
    df = df[:nCut]
    # Cut fully away largest IP (possibly partly cut now)
    m = (df.ip == df.ip.max())
    df = df[~m]

gc.collect(0)

df['sys'] = df.device * df.os.max() + df.os

df['cat_sys'] = pd.Categorical(df.sys)
df['system'] = df.cat_sys.cat.codes

df = df.drop(['sys', 'cat_sys'], axis = 1)


df['ip_and_sys'] = df.ip * df.system.max() + df.system


df['user'] = pd.Categorical(df.ip_and_sys)
df['user_code'] = df.user.cat.codes

df = df.drop(['ip_and_sys', 'user'], axis = 1)
df = df.drop(['ip', 'system'], axis = 1)

df = df.sort_values(by = ['user_code', 'time'])
df = df.reset_index()
df = df.drop(['index'], axis = 1)

import gc
gc.collect(0)


df.user_code.value_counts()



#
#
# Cluster analysis in TimeLineTool.py
#
#

d = TimeLineTool_analyse_user_code(df, 301)

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

