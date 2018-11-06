
import numpy as np
from datetime import datetime




def f(x):

    d = {}

    a = x['mjd'].values
    y = x['sampled_flux'].values

    p = x['passband'].values

    m_p0 = (p == 0)
    m_p1 = (p == 1)
    m_p2 = (p == 2)
    m_p3 = (p == 3)
    m_p4 = (p == 4)
    m_p5 = (p == 5)

    b = 300

    N = 200

    res = np.zeros((6, N), dtype = np.float32)

    a_min = np.min(a)
    a_max = np.max(a)

    L = a_max - a_min

    assert b < L

    begin = a_min + np.random.uniform(0, L - b)
    end = begin + b

    ai = a

    ai = ai - begin
    ai = ai / b
    ai = ai * N
    ai = ai + .5
    ai = ai.astype(dtype = np.int32)
    m = (ai >= 0) & (ai < N)

    d['num_cover'] = m.sum()

    d['num_unique'] = np.unique(ai[m]).shape[0]

    res[0, ai[m & m_p0]] = y[m & m_p0]
    res[1, ai[m & m_p1]] = y[m & m_p1]
    res[2, ai[m & m_p2]] = y[m & m_p2]
    res[3, ai[m & m_p3]] = y[m & m_p3]
    res[4, ai[m & m_p4]] = y[m & m_p4]
    res[5, ai[m & m_p5]] = y[m & m_p5]

    d['res']= res.flatten('F')

    return pd.Series(d)


def generate_sample_set(df):
    y_flux = df['flux'].values
    y_flux_err = df['flux_err'].values


    # In - sampled flux
    flux = np.random.normal(y_flux, y_flux_err)

    df = df.assign(sampled_flux = flux)

    start = datetime.now()

    r = df.groupby(['object_id']).apply(f)

    end = datetime.now()

    dT = end - start
    dSeconds = dT.total_seconds()

    print(f"Seconds {dSeconds}")

    s = r['res']

    data_sample = np.empty( (s.shape[0], 1200), dtype = np.float32)

    for idx, v in enumerate(s.values):
        data_sample[idx, :] = v

    return data_sample



data = generate_sample_set(df)
y_tot = y


for i in range(500):

    data = np.vstack([data, generate_sample_set(df)])
    y_tot= np.hstack([y_tot, y])


y = y_tot


0
Training until validation scores don't improve for 400 rounds.
[25]	valid_0's auc: 0.710426	valid_1's auc: 0.706768
[50]	valid_0's auc: 0.71669	valid_1's auc: 0.712744
[75]	valid_0's auc: 0.720878	valid_1's auc: 0.716675
[100]	valid_0's auc: 0.724129	valid_1's auc: 0.719866
[125]	valid_0's auc: 0.727134	valid_1's auc: 0.722596
[150]	valid_0's auc: 0.729583	valid_1's auc: 0.724782
[175]	valid_0's auc: 0.73167	valid_1's auc: 0.726657
[200]	valid_0's auc: 0.733558	valid_1's auc: 0.728331
[225]	valid_0's auc: 0.735449	valid_1's auc: 0.730051
[250]	valid_0's auc: 0.737282	valid_1's auc: 0.731692
