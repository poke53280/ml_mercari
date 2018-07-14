
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'train.csv')

def id_to_log (x):
    return np.log1p(int(x, base = 16))

id_log = train.ID.apply(id_to_log)

train = train.drop(['ID'], axis = 1)

def list_content(s):
    l = []
    for name, val in s.items():
        if val != 0:
            l.append(val)
    return l

def disp_row(s):
    l = list_content(s)
    a = np.array(l)
    unique, counts = np.unique(a, return_counts=True)
    d = dict(zip(unique, counts))

    d['Target'] = s['target']

    return d


def get_text_from_row(train, iRow):
    s = train.loc[[iRow]].iloc[0]

    l = list_content(s)

    txt = ""

    for x in l:
        clean = re.sub(r'\.', 'X', str(x))
        txt = txt + " " + clean

    return txt
"""c"""


anRows = []

for i in range(len(train)):
    txt = get_text_from_row(train, i)
    anRows.append(txt)

"""c"""

cv = CountVectorizer()

cv.fit(anRows)

X = cv.transform(anRows)

X = X.astype('float32')


def train(X, y):

    NUM_FOLDS = 5

    # Input to training:
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

    lKF = list (enumerate(kf.split(X)))

    lRMS = []

    y_oof = np.zeros(len (y))
    prediction = np.zeros(X.shape[0])

    while len(lKF) > 0:
        iLoop, (train_index, test_index) = lKF.pop(0)

        print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")
        
        X_train = X[train_index]
        y_train = y[train_index]
    
        X_valid = X[test_index]
        y_valid = y[test_index]

        rf = RandomForestRegressor(max_depth=70, n_estimators=400)

        rf.fit(X_train, y_train)

        y_p = rf.predict(X_valid)

        y_oof[test_index] = y_p

        rmsle_error = np.sqrt(mean_squared_error(y_p, y_valid))
        print(f"Rmsle: {rmsle_error}")

        lRMS.append(rmsle_error)


train (X, y)


train.ID



train['x'] = x



x = test.ID.apply(to_int)

test['x'] = x

test['T'] = 'test'
train['T'] = 'train'

df = pd.concat([train, test], axis = 0)

df = df.sort_values(by = 'x')

df[['x', 'T']]


