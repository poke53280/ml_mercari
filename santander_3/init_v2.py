
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

def id_to_int (x):
    return int(x, base = 16)

id_log = train.ID.apply(id_to_log)



#plot x, y
import matplotlib.pyplot as plt

plt.plot(x, y)

plt.show()



train = train.drop(['ID'], axis = 1)

def list_content(s):
    l = []
    for name, val in s.items():
        if val != 0:
            l.append(val)
    return l

def list_content_by_row(iRow):
    s = train.loc[[iRow]].iloc[0]
    return list_content(s)

def disp_row(s):
    l = list_content(s)
    a = np.array(l)
    unique, counts = np.unique(a, return_counts=True)
    d = dict(zip(unique, counts))

    d['Target'] = s['target']

    return d

def disp_row_by_row(iRow):
    s = train.loc[[iRow]].iloc[0]
    return disp_row(s)



def get_text_from_row(train, iRow):
    s = train.loc[[iRow]].iloc[0]

    l = list_content(s)

    txt = ""

    for x in l:
        clean = re.sub(r'\.', 'X', str(x))
        txt = txt + " " + clean

    return txt
"""c"""

float.as_integer_ratio(3.33333333)

from fractions import Fraction

Fraction(9.3333333333).limit_denominator()

f = 3.3333

Fraction(str(f))


disp_row_by_row(4000)

l = list_content_by_row(4000)


an = np.array(l)
an = an/ 1000.0

l = list (an)

################



def col_tracker_by_row(row):
    l = list_content_by_row(row)
    return col_tracker_by_list(l)

def col_tracker_by_col(c):
    l = list (train[col[c]].values)

    return col_tracker_by_list(l)


def is_col_candidate(x):
    x = x / 1000.0
    if x > 0.0 and x < 9999.9 and x == int (x):
        n = int(x)
        b = ((n//10) == (n/10))
        if b:
            pass
        else:
            return True

    return False


def col_tracker_by_list(l):

    an = np.array(l)
    an = an/ 1000.0

    l = list (an)

    num_zero = 0
    num_nzero = 0

    anRowCandidates = []

    num_nzero_in_range = 0

    for x in l:
        if x > 0.0 and x < 9999.9 and x == int (x):
            n = int(x)
            b = ((n//10) == (n/10))
            if b:
                num_zero = num_zero + 1
            else:
                num_nzero = num_nzero + 1
                anRowCandidates.append(n)

        else:
            pass

    print(f"All: {len(l)}. In range but zero ending: {num_zero}, in digit range non zero: {num_nzero}")

    anRowCandidates = np.array(anRowCandidates)

    if len(anRowCandidates) == 0:
        print(f"No col candidates")
    else:

        print(f"In range min-max: {anRowCandidates.min()} - {anRowCandidates.max()}")

        m5000 = (anRowCandidates < 5000)
        m4000 = (anRowCandidates < 4000)
        m3000 = (anRowCandidates < 3000)
        m1000 = (anRowCandidates < 1000)
        m100 = (anRowCandidates < 100)
        m10 = (anRowCandidates < 10)

        nB5000 = len (anRowCandidates[m5000])
        nB4000 = len (anRowCandidates[m4000])
        nB3000 = len (anRowCandidates[m3000])
        nB1000 = len (anRowCandidates[m1000])
        nB100 = len (anRowCandidates[m100])
        nB10 = len (anRowCandidates[m10])

        print(f"[< 5000: {nB5000}] [< 4000: {nB4000}] [< 3000: {nB3000}] [< 1000: {nB1000}] [< 100: {nB100}] [< 10: {nB10}]")

    return anRowCandidates

"""c"""

col_tracker_by_row(1005)



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


l = col_tracker(1183) # A lot of candidates, 118, none above


l


al = np.array(l)

np.unique(al)



1177   # 324 - no col candidates
1178  # 100 - no cand

1179 # 39 - no cand





1183 296: 119 in digit range, all below 5000

==> Many to 805, 14, and a few others



l = col_tracker(1184)

l

l = col_tracker(1185)

al = np.array(l)

np.unique(al)

l = col_tracker(1190)

l


l = col_tracker(1191)

l

col_attempts = []

for i in range(train.shape[0]):
    l = col_tracker_by_row(i)

    print (len(l))
    col_attempts = col_attempts + list (l)


import matplotlib.pyplot as pl

anCol = np.array(col_attempts)

anCol.min()
anCol.max()
anCol.mean()

pl.hist(anCol, bins= 100)

pl.show()

########
# Set up a text matrix. replace probable indices.

az.shape


dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])


dt = np.dtype(np.unicode_, 16)


az[:, :] = ""

az[8,4] = "C899"

az

train[train.columns[0]]


X = np.array(train)
az = np.zeros(train.shape, dtype = ('|U8'))


def set_txt_on_df_value(X, value, az, txt, isClearSource):
    m = (X == value)
    az[m] = txt

    if isClearSource:
        X[m] = 0

"""c"""


g_col_counter = 0
g_val_counter = 0

for process_col in range (0, X.shape[1]):

    t = X[:,process_col]
    values = np.unique(t)

    print(f"Unique items in col {process_col} is {len(values)}")

    nColCandidatesConverted = 0
    nValCandidatesConverted = 0

    for idx, v in enumerate(values):
        if is_col_candidate(v):
            # print(f"Is col candiate: {v}")

            txt = "C" + str(g_col_counter)
            set_txt_on_df_value(X, v, az, txt, True)
            g_col_counter = g_col_counter + 1
            nColCandidatesConverted = nColCandidatesConverted + 1

        else:
            txt = "V" + str(g_val_counter)
            set_txt_on_df_value(X, v, az, txt, True)
            g_val_counter = g_val_counter + 1
            nValCandidatesConverted = nValCandidatesConverted + 1
    
    print(f"    {nColCandidatesConverted} col candidates converted")         
    print(f"    {nValCandidatesConverted} val candidates converted")  

print(f" Col count is {g_col_counter} val count is {g_val_counter}")