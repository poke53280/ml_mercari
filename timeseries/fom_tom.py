


GRENSE_VED_OPPDELING = 31.0

fom_l = np.array([100, 130, 160, 190])
tom_l = np.array([133, 141, 225, 265])
l_ext_data = [31, 11, 24, -1]

df = pd.DataFrame({'ext_data': l_ext_data, 'fom': fom_l, 'tom': tom_l})

df = df.assign(sm_l = df.tom - df.fom + 1)



df = df.assign(antallDeler = np.ceil(df.sm_l / GRENSE_VED_OPPDELING))
df = df.assign(grunnlengde = np.floor(df.sm_l / df.antallDeler))
df = df.assign(rest = df.sm_l % df.grunnlengde)

df = df.loc[df.index.repeat(df.antallDeler)]

df = df.reset_index()

df = df.assign(local_idx = df.groupby('index').cumcount())

m_add = df.local_idx < df.rest

df = df.assign(m_add = m_add)

lengde = df.grunnlengde.copy()

lengde[m_add] = lengde[m_add] + 1

df = df.assign(lengde = lengde)

df = df.assign(acc_lengde = df.groupby('index').lengde.cumsum())

df = df.assign(tom_acc = df.fom + df.acc_lengde - 1)
df = df.assign(fom_acc = df.tom_acc - df.lengde + 1)

df[['index', 'ext_data', 'fom', 'tom', 'fom_acc', 'tom_acc']]




#################################################################################################


import numpy as np
from collections import Counter

def scatter_day(tom_acc, prior_weights):

    l_candidates = np.arange(len(prior_weights))

    add_n = np.random.choice(l_candidates, len(tom_acc), p=prior_weights)

    return tom_acc + add_n

def get_scatter_count(tom_acc, prior_weights):
    res = scatter_day(tom_acc0, prior_weights0)
    unique, counts = np.unique(res, return_counts=True)
    d = dict(zip(unique, counts))

    return d


tom_acc0 = [100, 112, 120, 112, 105, 109, 117, 121]
prior_weights0 = np.array([0.1, 0.3, 0.3, 0.2, 0.1])

tom_acc1 = [103, 114, 112, 122, 115, 119, 107, 111, 117]
prior_weights1 = np.array([0.1, 0.3, 0.1, 0.2, 0.1])


l_data = [(tom_acc0, prior_weights0), (tom_acc1, prior_weights1)]

A = Counter()

for tom, weights in l_data:
    d = get_scatter_count(tom, weights)
    A.update(d)


A



tom_acc = np.zeros(10000)

w = prior_weights

res = scatter_day(tom_acc, w)

diff = res - tom_acc

nAnalytical = w[0] * tom_acc.shape[0]

nSampled = res[diff == 0].shape[0]

print(nAnalytical, nSampled)