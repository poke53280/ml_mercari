

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc
from lightgbm import LGBMClassifier









DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)


filename_base = DATA_DIR + "df_t_xhu_Merged.pkl"



df = pd.read_pickle(filename_base)

gc.collect()

y = df['target_90']


gc.collect()

y = np.array(y, dtype = np.float16)

object_ID = df['object_id']

df = df.drop(['object_id', 'target_90'], axis = 1)

gc.collect()

data = df.values
data = data.astype(np.float32)


uniqueIDs = np.unique(object_ID)

num_splits = 8

folds = KFold(n_splits=num_splits, shuffle=True, random_state=11)

oof_preds = np.zeros(data.shape[0])

for n_fold, (trn_idx_unique, val_idx_unique) in enumerate(folds.split(uniqueIDs)):
    
    trn_uniques = uniqueIDs[trn_idx_unique]
    val_uniques = uniqueIDs[val_idx_unique]

    trn_idx = object_ID.isin(trn_uniques)
    val_idx = object_ID.isin(val_uniques)
    
    trn_x, trn_y = data[trn_idx], y[trn_idx]
    val_x, val_y = data[val_idx], y[val_idx]

    print (n_fold)

    clf = LGBMClassifier(n_estimators=199300, learning_rate=0.1, max_depth=4, num_leaves = 127, silent=-1, verbose=-1)


    clf.fit(trn_x, trn_y,  eval_set= [(trn_x, trn_y), (val_x, val_y)],
            eval_metric='auc', verbose=25, early_stopping_rounds=400)  

    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]

    print('Fold %2d AUC : %.3f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()

print('Full AUC score %.3f' % roc_auc_score(y, oof_preds)) 




#  1000 x 2 : AUC 0.644554
# 10,000 x 2: AUC 0.674902

# 1000 wide: AUC 0.728738  likely overtraining + leak.

# Warning. Leaking: Several sampled rows of same object. Training on same object as found in validation.

# 10,000 wide: AUC ~ 0.81 but likely large leak. valid score close to one.

# 100 wide x2 no leak 0.663800

# 1000 no leak AUC 0.709835 w overtraining


# 10000 wide no leak
# Full AUC score 0.746
# 19300, stop 100
# Full AUC score 0.747
# "29300""

#200,000 x2 super wide no leak

# 0.782, 0.762


# New dataset (fast, sampling all bands)

# AUC 0.6

# New dataset fixed bug 200,000 per 1/3. Trained on 1/3
#
# AUC 0.88
#

# 2/ 3 very similar

 







