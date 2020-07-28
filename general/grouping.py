


import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy.stats import norm


import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

b = np.array([18268, 18276, 18281, 18290, 18298, 18310, 18319, 18330, 18339])
e = np.array([18268, 18276, 18290, 18295, 18299, 18310, 18319, 18330, 18339])

df = pd.DataFrame({ 'fom':b,'tom':e})

df = pd.concat([df, df, df], axis = 0, ignore_index= True)

l_id = [0] * 9 + [1] * 9 + [2] * 9

df = df.assign(id = l_id)



df['range'] = [list(range(i, j+1)) for i, j in df[['fom', 'tom']].values]
               
df

id = np.array([0,0,0])

b = np.array([18220, 18229, 18251])
e = np.array([18250, 18245, 18260])

df = pd.DataFrame({'id': id, 'fom':b, 'tom':e})


df = df.sort_values(by = 'fom').reset_index(drop = True)


e_below = np.roll(df.tom, -1)
id_below = np.roll(df.id, -1)

df = df.assign(end_next = e_below)
df = df.assign(id_next = id_below)

below_short = (df.end_next <= df.tom) & (df.id_next == df.id)


gap = df.fom_next - df.tom - 1

df = df.assign(gap = gap)

m_new = (df.gap > 3) | (df.id_next != df.id)

i_new = m_new.astype(int)

i_new = np.roll(i_new, 1)
i_new[0] = 0

idx_group = np.cumsum(i_new)

df = df.assign(idx_group = idx_group)

g = df.groupby('id').idx_group.min()

min_id = df.id.map(g)

idx_group = idx_group - min_id

df = df.assign(idx_group = idx_group)

df = df.drop(['fom_next', 'id_next', 'gap'], axis = 1)




data = np.array([8, 9, 10, 4, 6, 10, 12])


np.diff(data)

cumsum_diff = np.cumsum(np.diff(data))

cumsum_diff[1:] >= cumsum_diff[:-1]