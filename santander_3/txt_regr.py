
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'txt_db_huge.csv')

train.columns= ['1', 'txt', 'target']
train = train.drop(['1'], axis = 1)

y = train.target

train = train.drop('target', axis = 1)


from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(ngram_range = (1,2))
NUM_FOLDS = 5

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

lKF = list (enumerate(kf.split(train)))

lRMS = []

y_oof = np.zeros(len (y))
prediction = np.zeros(train.shape[0])


while len(lKF) > 0:
    iLoop, (train_index, test_index) = lKF.pop(0)

    print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")

    X_train = train.txt[train_index]
    y_train = y[train_index]
    
    X_valid = train.txt[test_index]
    y_valid = y[test_index]

    Xt = cv.fit_transform(X_train)
    Xv = cv.transform(X_valid)
    
    Xt = Xt.astype(np.float32)
    Xv = Xv.astype(np.float32)

    l = LGBMTrainer_BASIC()

    l.train_with_validation(Xt, y_train, Xv, y_valid)

    y_p = l.predict(Xv)

    rmsle_error = np.sqrt(metrics.mean_squared_error(y_p, y_valid))
    print(f"Rmsle: {rmsle_error}")
    lRMS.append(rmsle_error)

anRMS = np.array(lRMS)
RMSLEmean = anRMS.mean()
RMSLEstd  = anRMS.std()

print(f"  ==> RMSLE = {RMSLEmean} +/- {RMSLEstd}")
print(" ------------------------------------------------- ")