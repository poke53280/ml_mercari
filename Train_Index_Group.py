

import pandas as pd
import numpy as np



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
