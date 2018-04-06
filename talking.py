

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import datetime
import pylab as pl

##############################################################################
#
#        Close1D
#

def Close1D(n, threshold):

    nID = 0

    e = np.zeros(len (n), dtype = np.int)

    last_value = -1     # impossible value

    for idx, x in np.ndenumerate(n):
        if last_value == -1:
            pass
        else:       
            diff = x - last_value

            if diff <= threshold:
               pass
            else:
                nID = nID + 1

        e[idx] = nID
    
        last_value = x

    return e

"""c"""



DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\talking\\"            
DATA_DIR_BASEMENT = "c:\\data_talking\\"
DATA_DIR = DATA_DIR_PORTABLE


print('loading train data...')
df = pd.read_csv(DATA_DIR + "train.csv")


# Reduce dataset to contain all on a few IPs.

# Approx two minutes sorting

df = df.sort_values(by = ['ip'])

len(df)
# 184,903,890


CUT_TO_ENTRIES = 10000000   # About 1/18 of full set



df = df[:CUT_TO_ENTRIES]

# Cut away largest IP (possibly chopped in session)

m = (df.ip == df.ip.max())
df = df[~m]


df['epoch'] = (pd.to_datetime(df.click_time) - datetime.datetime(1970, 1, 1))

df['pim'] = df.epoch/ np.timedelta64(1, 's')
df['pim2'] = df.pim.astype(np.int64)

df = df.drop(['epoch', 'pim', 'click_time'], axis = 1)

MIN_EPOCH = df.pim2.min()

df['time'] = df.pim2 - MIN_EPOCH

df = df.drop(['pim2'], axis = 1)
df = df.drop(['is_attributed'], axis = 1)

df['attrib'] = (pd.to_datetime(df.attributed_time) - datetime.datetime(1970, 1, 1))
df = df.drop(['attributed_time'], axis = 1)

df['pim'] = df.attrib/ np.timedelta64(1, 's')
df = df.drop(['attrib'], axis = 1)

df.pim = df.pim.fillna(0)

df['pim2'] = df.pim.astype(np.int64)
df = df.drop(['pim'], axis = 1)


def hello4(x):
    if x > 0:
        return x - MIN_EPOCH
    else:
        return x

"""c"""

q = df.pim2.apply(hello4)

df = df.assign(a_time = q)


m = df.pim2 > 0
df[m]

df = df.drop(['pim2'], axis = 1)


df['sys'] = df.device * df.os.max() + df.os
df['cat_sys'] = pd.Categorical(df.sys)
df['system'] = df.cat_sys.cat.codes

df = df.drop(['sys', 'cat_sys'], axis = 1)


df['ip_and_sys'] = df.ip * df.system.max() + df.system


df['user'] = pd.Categorical(df.ip_and_sys)

df['user_code'] = df.user.cat.codes

df = df.drop(['ip_and_sys', 'user'], axis = 1)

df = df.drop(['ip', 'system'], axis = 1)

df['time'] = pd.to_numeric(df.time, downcast = 'integer')
df['a_time'] = pd.to_numeric(df.a_time, downcast = 'integer')

df = df.sort_values(by = ['user_code', 'time'])

df = df.reset_index()
df = df.drop(['index'], axis = 1)



lcNCluster = []

print(f"#Values: {len(q)}")

for threshold in lcThreshold:

    e = Close1D(acData, threshold)
    nClusters = e.max() + 1

    fClusterSize = len(q)/nClusters

    lcNCluster.append(nClusters)

    print(f"Threshold {threshold} sec: nClusters = {nClusters}. Values per cluster: {fClusterSize:.1f}")

"""c"""


pl.plot(lcThreshold,lcNCluster)
pl.xlabel('Slack (secs)')
pl.ylabel('Clusters')
pl.title('# clusters' + str(test_id))
pl.show()


SESSION_THRESHOLD = 60

user_code = np.array(df.user_code)
click_time = np.array(df.time)

res = np.empty(len (user_code), dtype = np.int)


for u in range(user_code.min(), user_code.max() + 1):
    begin = np.searchsorted(user_code, u)
    end   = np.searchsorted(user_code, u+1)

    print(f"For user_code {u}: start index = {begin}, beyond end={end}")

    e = Close1D(click_time[begin:end], SESSION_THRESHOLD)

    if begin == 0:
        pass
    else:
        max_used = np.max(res[:begin])
        e = e + max_used +1 

    res[begin:end] = e

"""c"""    

df = df.assign(session=res)

df = df.drop(['user_code'], axis = 1)


q = df [:90]
groups = df.groupby(q.session)
groups.groups


pd.pivot_table(q,index=["session"])...



