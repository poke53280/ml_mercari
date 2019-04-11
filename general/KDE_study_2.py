

import numpy as np
from sklearn.neighbors.kde import KernelDensity

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import pandas as pd



def cluster_sm(b, e, kde, prob_threshold, L_threshold):

    sm = np.column_stack([b, e])

    sm[:, 1] += 1

    s = set()

    for i in range (sm.shape[0]):
        s = s.union(range(sm[i][0], sm[i][1]))
    """c"""

    x = np.array(list(s)).reshape(-1,1)

    x_min = np.min(x)
    x_max = np.max(x)

    kde.fit(x)

    x_d = np.array(range(x_min, x_max + 1))

    logprob = kde.score_samples(x_d[:, None])

    prob = np.exp(logprob)

    m = prob > prob_threshold

    if m.sum() == 0:
        return (-1, -1, -1)

    x_s = x_d[m]

    i_start = np.where(np.diff(x_d[m]) > 1)[0]

    i_start = i_start + 1
    i_start = np.insert(i_start, 0, 0)

    i_end = i_start[1:]
    i_end = np.append(i_end, x_s.shape[0])


    x_lo = x_s[i_start]
    x_hi = x_s[i_end -1]

    m = (x >= x_lo) & (x <= x_hi)


    g_start = np.empty(m.shape[1], dtype = np.int32)
    g_end   = np.empty(m.shape[1], dtype = np.int32)
    g_L     = np.empty(m.shape[1], dtype = np.int32)

    for i in range(m.shape[1]):
        g_i =  x[m[:, i]]
        g_start[i] = np.min(g_i)
        g_end[i] = np.max(g_i)
        g_L[i] = g_i.shape[0]

    """c"""

    i = np.where(g_L >= L_threshold)[0]

    start = -1
    end = -1
    L = -1

    if i.shape[0] == 0:
        pass
    else:
        idx = np.max(i)
        start = g_start[idx]
        end = g_end[idx]
        L = g_L[idx]
    """c"""

    return (start, end, L)
"""c"""


kde = KernelDensity(bandwidth= 1.0, kernel = 'gaussian')


l = []

id_min = 0
id_max = 2000

n_entries = 10000

for n in range(n_entries):

    id = np.random.random_integers(0, id_max, 1)[0]

    start =  np.random.random_integers(0, 10000, 1)[0]
    end = start + np.random.random_integers(0, 100, 1)[0]
    reg = start + np.random.random_integers(0, 10, 1)[0]
    ext =  np.random.random_integers(0, 100, 1)[0]


    l.append((id, start, reg, end, ext))
"""c"""


df = pd.DataFrame(l)

df.columns = ['id', 'begin', 'reg', 'end', 'ext']


L_threshold = 7

prob_threshold = 0.001

df_res = generate(df, L_threshold, prob_threshold)

df_res


def generate(df, L_threshold, prob_threshold):

    anID = np.unique(df.id.values)

    anBegin = np.array(df.begin)
    anEnd   = np.array(df.end)
    enExt   = np.array(df.ext)

    anB = np.empty_like(anBegin)
    anL = np.empty_like(anBegin)

    for id in anID:

        m = df.id.values == id

        b, e, l = cluster_sm(anBegin[m], anEnd[m], kde, prob_threshold, L_threshold)

        anB[m] = b
        anL[m] = l
    """c"""

    # Only keep information known at or before L_threshold days into leave.
    # That is registration date must be maximum beginning of leave + L_threshold days

    m = (anL >= 0) & ((df.reg.values - anB) < L_threshold)

    df = df[m].reset_index(drop = True)

    y = anL[m] - L_threshold

    df = df.assign(Y = y)

    anID = np.unique(df.id.values)

    begin = df.begin.values
    reg = df.reg.values
    end = df.end.values

    l = end - begin + 1

    df = df.assign(l = l)

    df = df.drop(['end'], axis = 1)

    t_max = np.empty_like(begin)

    for id in anID:
        m = df.id.values == id
        t_max[m] = np.max([np.max(begin[m]),np.max(reg[m])]) 
    """c"""

    df = df.assign(t_max = t_max)


    reg_delta = df.reg - df.begin

    df = df.assign(d_reg = reg_delta)

    df = df.drop(['reg'], axis = 1)

    b = - (df.begin - df.t_max)

    df = df.assign(begin = b)

    df = df.sort_values(by = ['id', 'begin'])
    df = df.reset_index(drop = True)


    b_step = np.empty_like(b)

    anID = np.unique(df.id.values)

    for id in anID:
        m = df.id.values == id

        b = df.begin[m].values

        b_step_local = np.insert(np.diff(b), 0, b[0])

        b_step[m] = b_step_local


    df = df.assign(begin = b_step)


    B_str = ('B' + df.begin.astype(str)).values
    E_str = ('E' + df.ext.astype(str)).values
    L_str = ('L' + df.l.astype(str)).values
    R_str = ('R' + df.d_reg.astype(str)).values

    anID = np.unique(df.id.values)

    t_max = []
    t_Y = []
    t_s = []

    for id in anID:
        m = df.id.values == id

        c = np.empty( 4 * B_str[m].shape[0], dtype=object)

        c[0::4] = B_str[m]
        c[1::4] = E_str[m]
        c[2::4] = L_str[m]
        c[3::4] = R_str[m]

        s = np.array_str(c)

        s = s.replace("'", "")
        s = s.replace("[", "")
        s = s.replace("]", "")

        t_s.append(s)
        t_Y.append(df.Y[m].values[0])
        t_max.append(df.t_max[m].values[0])
    """c"""

    df_res = pd.DataFrame({'t': t_max, 's': t_s, 'y': t_Y})

    return df_res
"""c"""





