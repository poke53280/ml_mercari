
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer


from santander_3.lgbm_basic import LGBMTrainer_BASIC

def local_load_txt_regr():
    DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
    DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
    DATA_DIR = DATA_DIR_PORTABLE

    train = pd.read_csv(DATA_DIR + 'txt_db_basic.csv')

    train.columns= ['1', 'target', 'txt']
    train = train.drop(['1'], axis = 1)

    y = train.target

    train = train.drop('target', axis = 1)


def txt_reg(data, y):

    cv = CountVectorizer(ngram_range = (1,2))
    NUM_FOLDS = 10

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

    lKF = list (enumerate(kf.split(data)))

    lRMS = []
    
    prediction = np.zeros(len (y))


    while len(lKF) > 0:
        iLoop, (train_index, test_index) = lKF.pop(0)

        print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")

        X_train = data.txt[train_index]
        y_train = y[train_index]
    
        X_valid = data.txt[test_index]
        y_valid = y[test_index]

        Xt = cv.fit_transform(X_train)
        Xv = cv.transform(X_valid)
    
        Xt = Xt.astype(np.float32)
        Xv = Xv.astype(np.float32)

        l = LGBMTrainer_BASIC()

        print(f"Train X {Xt.shape}")

        l.train_with_validation(Xt, y_train, Xv, y_valid)

        y_p = l.predict(Xv)

        rmsle_error = np.sqrt(metrics.mean_squared_error(y_p, y_valid))
        print(f"Rmsle: {rmsle_error}")
        lRMS.append(rmsle_error)

    anRMS = np.array(lRMS)
    RMSLEmean = anRMS.mean()
    RMSLEstd  = anRMS.std()
    print(f"  ==> RMSLE = {RMSLEmean} +/- {RMSLEstd}")
    return (RMSLEmean, RMSLEstd)

"""c"""
