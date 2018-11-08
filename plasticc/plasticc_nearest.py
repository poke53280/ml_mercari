
import numpy as np
from datetime import datetime
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import gc


DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

df_meta = pd.read_csv(DATA_DIR + "training_set_metadata.csv")
df = pd.read_csv(DATA_DIR + "training_set.csv")

y = np.array (df_meta.target, dtype = np.int32)


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

    res = np.empty((6, N), dtype = np.float32)
    res[:, :] = np.nan

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

num_sets = 500

for i in range(num_sets):
    print(f"Generating {i+1}/ {num_sets}")
    data = np.vstack([data, generate_sample_set(df)])
    y_tot= np.hstack([y_tot, y])

y = y_tot

m = y == 90

y[m] = 1
y[~m] = 0


num_tot = y.shape[0]

num_pos = m.sum()
num_neg = num_tot - num_pos

r_scale_pos = num_neg / num_pos

num_splits = 8

folds = KFold(n_splits=num_splits, shuffle=True, random_state=11)

oof_preds = np.zeros(data.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
        
    trn_x, trn_y = data[trn_idx], y[trn_idx]
    val_x, val_y = data[val_idx], y[val_idx]

    print (f"Fold {n_fold + 1} / {num_splits}")

    clf = LGBMClassifier(n_estimators=20000, learning_rate=0.03, max_depth = 7, silent=-1, verbose=-1, scale_pos_weight = r_scale_pos)

    clf.fit(trn_x, trn_y,  eval_set= [(trn_x, trn_y), (val_x, val_y)], eval_metric='auc', verbose=10, early_stopping_rounds=400)  

    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]

    print('Fold %2d AUC : %.3f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()


print('Full AUC score %.3f' % roc_auc_score(y, oof_preds)) 







# On + 500  pre nan, balance.
# LGBMClassifier(n_estimators=20000, learning_rate=0.01, num_leaves = 255, silent=-1, verbose=-1)
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

# on + 50
[825]	valid_0's auc: 0.765444	valid_1's auc: 0.716776
[850]	valid_0's auc: 0.766402	valid_1's auc: 0.717239
[875]	valid_0's auc: 0.767495	valid_1's auc: 0.71753
[900]	valid_0's auc: 0.768663	valid_1's auc: 0.71784
[925]	valid_0's auc: 0.76968	valid_1's auc: 0.718197
[950]	valid_0's auc: 0.770798	valid_1's auc: 0.718658



# on + 500
#LGBMClassifier(n_estimators=20000, learning_rate=0.03, max_depth = 7, silent=-1, verbose=-1, scale_pos_weight = r_scale_pos)
[25]	valid_0's auc: 0.610846	valid_1's auc: 0.609485
[50]	valid_0's auc: 0.645241	valid_1's auc: 0.64345
[75]	valid_0's auc: 0.659995	valid_1's auc: 0.657281
[100]	valid_0's auc: 0.667646	valid_1's auc: 0.664632
[125]	valid_0's auc: 0.673534	valid_1's auc: 0.67054
[150]	valid_0's auc: 0.679768	valid_1's auc: 0.676866
[175]	valid_0's auc: 0.688182	valid_1's auc: 0.68519
[200]	valid_0's auc: 0.693115	valid_1's auc: 0.69031
[225]	valid_0's auc: 0.698678	valid_1's auc: 0.695511
[250]	valid_0's auc: 0.703199	valid_1's auc: 0.700051
[275]	valid_0's auc: 0.706202	valid_1's auc: 0.702894
[300]	valid_0's auc: 0.708985	valid_1's auc: 0.705541
[325]	valid_0's auc: 0.711022	valid_1's auc: 0.707315
[350]	valid_0's auc: 0.712954	valid_1's auc: 0.709161
[375]	valid_0's auc: 0.714698	valid_1's auc: 0.710693
[400]	valid_0's auc: 0.716218	valid_1's auc: 0.712025
[425]	valid_0's auc: 0.717547	valid_1's auc: 0.713193
[450]	valid_0's auc: 0.718957	valid_1's auc: 0.714504
[475]	valid_0's auc: 0.720101	valid_1's auc: 0.715403
[500]	valid_0's auc: 0.721142	valid_1's auc: 0.716313
[525]	valid_0's auc: 0.722369	valid_1's auc: 0.717344
[550]	valid_0's auc: 0.723403	valid_1's auc: 0.718208
[575]	valid_0's auc: 0.724271	valid_1's auc: 0.718871
[600]	valid_0's auc: 0.725141	valid_1's auc: 0.719642
[625]	valid_0's auc: 0.725859	valid_1's auc: 0.72029
[650]	valid_0's auc: 0.726649	valid_1's auc: 0.720915
[675]	valid_0's auc: 0.727437	valid_1's auc: 0.72159
[700]	valid_0's auc: 0.728208	valid_1's auc: 0.722262
[725]	valid_0's auc: 0.728855	valid_1's auc: 0.722802
[750]	valid_0's auc: 0.729543	valid_1's auc: 0.723388
[775]	valid_0's auc: 0.730177	valid_1's auc: 0.723935


# on + 500 same as old:
[10]	valid_0's auc: 0.699383	valid_1's auc: 0.696548
[20]	valid_0's auc: 0.705956	valid_1's auc: 0.702758
[30]	valid_0's auc: 0.709452	valid_1's auc: 0.705933
[40]	valid_0's auc: 0.711771	valid_1's auc: 0.708221
[50]	valid_0's auc: 0.714044	valid_1's auc: 0.710468
[60]	valid_0's auc: 0.71573	valid_1's auc: 0.712051
[70]	valid_0's auc: 0.717299	valid_1's auc: 0.713439
[80]	valid_0's auc: 0.718761	valid_1's auc: 0.714747
[90]	valid_0's auc: 0.720308	valid_1's auc: 0.716098
[100]	valid_0's auc: 0.721582	valid_1's auc: 0.717329
[110]	valid_0's auc: 0.722907	valid_1's auc: 0.718582
[120]	valid_0's auc: 0.724117	valid_1's auc: 0.71971
[130]	valid_0's auc: 0.725249	valid_1's auc: 0.720735
[140]	valid_0's auc: 0.726342	valid_1's auc: 0.72175
[150]	valid_0's auc: 0.727436	valid_1's auc: 0.722683
[160]	valid_0's auc: 0.728543	valid_1's auc: 0.723647
[170]	valid_0's auc: 0.729661	valid_1's auc: 0.724589
[180]	valid_0's auc: 0.730673	valid_1's auc: 0.725539
[190]	valid_0's auc: 0.731646	valid_1's auc: 0.726436
[200]	valid_0's auc: 0.732543	valid_1's auc: 0.72727
[210]	valid_0's auc: 0.733444	valid_1's auc: 0.728074
[220]	valid_0's auc: 0.734327	valid_1's auc: 0.728849


