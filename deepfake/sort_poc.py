

import numpy as np
import pandas as pd

num_id = 30000000
id = np.arange(num_id)


num_data = 33000000

d_id = np.arange(num_id)
d_id_extra = np.random.choice(num_id, replace = True, size = num_data - num_id)

d_id = np.concatenate([d_id, d_id_extra])

ad_id = np.sort(d_id)

sID = pd.Series(ad_id)
anData = np.random.choice(50, size = sID.shape[0])

df_d = pd.DataFrame({'id' : sID, 'data': anData})


id = np.arange(num_id)

anLo = np.searchsorted(ad_id, id, side = 'left')
anHi = np.searchsorted(ad_id, id, side = 'right')

df = pd.DataFrame({'id': id, 'lo': anLo, 'hi': anHi})
df = df.assign (L = anHi - anLo)


data0 = pd.Series(np.zeros(df.shape[0]) - 1)
m = L > 0
idx0 = df.lo[m] + 0
data0[m] = anData[idx0]

data1 = pd.Series(np.zeros(df.shape[0]) - 1)
m = L > 1
idx1 = df.lo[m] + 1
data1[m] = anData[idx1]

data2 = pd.Series(np.zeros(df.shape[0]) - 1)
m = L > 2
idx2 = df.lo[m] + 2
data2[m] = anData[idx2]

df = df.assign(data0 = data0, data1 = data1, data2 = data2)



id = 29999999

df[df.id == id]

df_d[df_d.id == id]







data0 = df.apply(lambda x: list(anData[x['lo']: x['hi']]), axis = 1)

df = df.assign(data = l_data)

np.arange(anLo, anHi)

indexes = 
anData[anLo[9]:anHi[9]]

anLo[9]


df

df_m = df.merge(df_d, how = 'inner', on = 'id')