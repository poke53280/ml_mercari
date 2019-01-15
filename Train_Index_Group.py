

import pandas as pd
import numpy as np


id = [0,0,1,1,3, 0]
d = [3,3,4,5,6, 3]
s = [4,4,4,4,4, 4]
ix = [7,2, 0, 3, 1, 4]

t = ['A', 'B', 'C', 'D', 'E', 'C']

df = pd.DataFrame({'id': id, 'd' : d, 's': s, 'idx': ix, 't':t})

df

# Group by id, d, s. Check t ordering.

df_grouped = df.groupby(['id', 'd', 's'])

for group_key, item in df_grouped:
    print(group_key)
    print(item)
"""c"""


g_id = []
g_d = []
g_s = []
g_t = []

for key, item in df_grouped:

    print (key)

    g = df_grouped.get_group(key)

    l = list (g.t)

    s = ".".join(l)

    g_id.append(key[0])
    g_d.append(key[1])
    g_s.append(key[2])

    g_t.append(s)
    
"""c"""

df_g = pd.DataFrame({'id': g_id, 'd':g_d, 's': g_s, 't' : g_t})

df_g

for x in g:
    print (x.index)
    print (x.names)





def full_sum(df):
    l_b = list()

    for x in df.b:
        l_b.append(x)

    l_c = list()

    for x in df.c:
        l_c.append(x)

    t = list (tuple(zip (l_b, l_c)))

    return t
"""c"""


num_objects = 90


df = pd.DataFrame({'id': np.random.choice(range(num_objects), 1000),
                   'a': np.random.randn(1000),
                   'b': np.random.randn(1000),
                   'c': np.random.randn(1000),
                   'd': np.random.randn(1000),
                   'N': np.random.randint(100, 1000, (1000))})



g = df.groupby('id')

g.agg(['sum'])


q = df[df.id == 89]
