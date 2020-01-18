
import dask
import numpy as np
import pandas as pd
import dask.dataframe as dd
import datetime
import numba as nb

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth= 540)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

@nb.jit(nb.types.Array(nb.int32, 0, "C")(nb.types.Array(nb.int32, 0, "C"), nopython = True)
def execute_numba(x):

    nSpace = 2
    
    x_d = np.arange(np.min(x) - nSpace, np.max(x) + nSpace + 1, dtype = np.float32)

    threshold_density = 1

    x = x.reshape(-1, 1)
    
    a = ((x_d > x - 8) & (x_d < x + 8)) * np.float32(1.0)
    b = ((x_d == x - 8) | (x_d == x + 8)) * np.float32(0.5)

    c = a + b

    density = np.sum(c, axis = 0)

    m = density >= threshold_density

    x_s = x_d[m]

    i_start = np.where(np.diff(x_s) > 1)[0]
    i_start = i_start + 1
    i_start = np.insert(i_start, 0, 0)

    i_end = i_start[1:]
    i_end = np.append(i_end, x_s.shape[0])

    x_lo = x_s[i_start]
    x_hi = x_s[i_end -1]

    m = (x >= x_lo) & (x <= x_hi)

    assert (m.sum(axis = 1) == 1).all(), "(m.sum(axis = 1) == 1).all()"

    g = np.where(m)[1]

    g_i = np.argsort(g)

    x = x[g_i]
    g = g[g_i]

    g_start_idx = np.where(np.diff(g) > 0)[0]
    g_start_idx = g_start_idx + 1
    g_start_idx = np.insert(g_start_idx, 0, 0)

    g_start = np.minimum.reduceat(x, g_start_idx)
    g_end = np.maximum.reduceat(x, g_start_idx)
    g_L = np.unique(g, return_counts = True)[1].reshape(-1, 1)

    g_full_length = g_end - g_start

    anMatrix = np.hstack([g_start, g_end, g_L])

    l = anMatrix.reshape(1, -1).ravel()

    return l
"""c"""


num_rows = 200000

num_ids = 40000
num_fom = 10000
num_tom = 100


id = np.random.choice(num_ids, num_rows)

d0 = np.random.uniform(size = num_rows)
fom = np.random.choice(num_fom, size = num_rows)
tom = np.random.choice(num_tom, size = num_rows)

tom = fom + tom

df = pd.DataFrame({'id': id, 'd0': d0, 'fom': fom, 'tom': tom})
df = df.sort_values(by = 'id').reset_index(drop = True)


z = list(zip(df.fom, df.tom))

z = pd.Series(z)

l = z.map(lambda x: list (range(x[0], x[1] + 1)))

df = df.assign(l = l)

df_g = df.groupby('id').l.sum()

xDays = df_g.map(lambda x: np.array(np.unique(x), dtype = np.int32))

df_g = pd.DataFrame(df_g)

df_g = df_g.assign(xDays = xDays)

ddf = dd.from_pandas(df_g, npartitions=16)

r = ddf.xDays.map(execute_numba, meta=pd.Series()).compute()

df_g = df_g.assign(r = r)

df_g = df_g.drop(['l', 'xDays'], axis = 1)





