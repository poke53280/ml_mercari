

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error



import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


import catboost

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)


df = pd.read_csv("C:\\users\\t149900\\source\\repos\\sm_data\\sm_data\\radon.txt")


def lgbm_short(df):
    X, y, c = preprocess(df)

    c = c.astype('category').cat.codes
    df = pd.DataFrame({'c' : c})
    df = df.assign(x = X)

    df = df.assign(y = y)

    df_train, df_valid = train_test_split(df, random_state = 43)

    x_col = [x for x in list(df.columns) if x != 'y']

    X_train = df_train[x_col]
    y_train = df_train.y

    X_valid = df_valid[x_col]
    y_valid = df_valid.y


    d_train = lgb.Dataset(X_train, label = y_train)
    d_valid = lgb.Dataset(X_valid, label = y_valid)

    model = lgb.LGBMRegressor(iterations=12, learning_rate=0.1, depth=10, cat_features = ['x', 'c'])

    model.fit(X_train, y_train)

    y_p = model.predict(X_valid)


    rmse = np.sqrt(mean_squared_error(y_valid, y_p))

    print(f"RMSE = {rmse}")




def lgbm(df):

    df = df.drop(['Unnamed: 0'], axis = 1)
    df = df.drop(['stfips'], axis = 1)
    df = df.drop(['idnum'], axis = 1)
    df = df.drop(['windoor'], axis = 1)
    df = df.drop(['state'], axis = 1)
    df = df.drop(['state2'], axis = 1)


    l_nominal = ['Uppm', 'fips', 'cntyfips', 'adjwt', 'pcterr', 'activity', 'stopdt', 'startdt', 'stoptm', 'starttm', 'wave']

    cols = df.columns

    l_cat = [x for x in cols if x not in l_nominal and x != 'log_radon']


    for c in l_cat:
        s = df[c].astype('category')
        df = df.assign(**{c: s})
    

    df_train, df_valid = train_test_split(df, random_state = 43)

    x_col = [x for x in list(df.columns) if x != 'log_radon']

    X_train = df_train[x_col]
    y_train = df_train.log_radon

    X_valid = df_valid[x_col]
    y_valid = df_valid.log_radon


    d_train = lgb.Dataset(X_train, label = y_train)
    d_valid = lgb.Dataset(X_valid, label = y_valid)

    model = lgb.LGBMRegressor(iterations=12, learning_rate=0.1, depth=3, cat_features = l_cat)

    model.fit(X_train, y_train)

    y_p = model.predict(X_valid)


    rmse = np.sqrt(mean_squared_error(y_valid, y_p))

    print(f"RMSE = {rmse}")




def preprocess(df):
    df = df[['basement', 'county', 'log_radon']]

    m_valid_basement = (df['basement'] == 'Y') | (df['basement'] == 'N')

    df = df[m_valid_basement].reset_index(drop = True)

    n_basement = np.empty(df.shape[0], dtype = np.float32)

    m_basement = df.basement == 'Y'

    n_basement[m_basement] = 1.0
    n_basement[~m_basement] = 0.0

    X = n_basement.reshape(-1, 1)
    y = df['log_radon']

    return X, y, df['county']




def flat_regression_all(df):
    
    X, y, _ = preprocess(df)
    l_RMSE = []

    kf = KFold(n_splits = 15, random_state = 3, shuffle = True)

    for tr_idx, va_idx in kf.split(X):
        X_train = X[tr_idx]
        X_valid = X[va_idx]
        y_train = y[tr_idx]
        y_valid = y[va_idx]

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_p = lr.predict(X_valid)

        mse = mean_squared_error(y_valid, y_p)
        rmse = np.sqrt(mse)
        print (f"RMSE = {rmse}")
        l_RMSE.append(rmse)


    anRMSE = np.array(l_RMSE)
    print(f"RMSE = {anRMSE.mean():.2f} +/- {anRMSE.std():.2f}")


           
def flat_regression_by_county_0(df):
    X, y, c = preprocess(df)

    l_RMSE = []

    kf = KFold(n_splits = 15, random_state = 3, shuffle = True)

    tr_idx, va_idx =  list(kf.split(X))[0]

    for tr_idx, va_idx in kf.split(X):
        X_train = X[tr_idx]
        X_valid = X[va_idx]
        y_train = y[tr_idx]
        y_valid = y[va_idx]

        c_train = c[tr_idx]
        c_valid = c[va_idx]

        y_p = np.zeros(X_valid.shape[0], dtype = np.float32)

        for county in np.unique(c_train):
            print (county)
            m = c_train == county

            lr = LinearRegression()
            lr.fit(X_train[m], y_train[m])

            m_p = c_valid == county

            if m_p.sum() > 0:
                y_p[m_p] = lr.predict(X_valid[m_p])


        mse = mean_squared_error(y_valid, y_p)
        rmse = np.sqrt(mse)
        print (f"RMSE = {rmse}")
        l_RMSE.append(rmse)

    anRMSE = np.array(l_RMSE)
    print(f"RMSE = {anRMSE.mean():.2f} +/- {anRMSE.std():.2f}")




def flat_regression_by_county_1(df):
    X_, y_, c_ = preprocess(df)

    l_RMSE_acc = []
    l_num = []

    y_mean = np.mean(y_)

    
    for county in np.unique(c_):
        m = c_ == county

        l_num.append(m.sum())

        X = X_[m]
        y = y_[m].reset_index(drop = True)

        l_RMSE = []

        n_splits = np.min([m.sum(), 5])

        if m.sum() == 1:
            rmse = np.sqrt(mean_squared_error([y[0]], [y_mean]))
            l_RMSE_acc.append(rmse)
            continue

        kf = KFold(n_splits = n_splits, random_state = 3, shuffle = True)

        tr_idx, va_idx =  list(kf.split(X))[0]

        for tr_idx, va_idx in kf.split(X):
            X_train = X[tr_idx]
            X_valid = X[va_idx]
            y_train = y[tr_idx]
            y_valid = y[va_idx]

            lr = LinearRegression()
            lr = lr.fit(X_train, y_train)

            y_p = lr.predict(X_valid)

            rmse = np.sqrt(mean_squared_error(y_valid, y_p))

            l_RMSE.append(rmse)

        anRMSE = np.array(l_RMSE)

        rmse = anRMSE.mean()
        rmse_std = np.std(anRMSE)

        print(f"{county}: {m.sum()} : RMSE = {rmse:.2f} +/- {rmse_std:.2f}")

        l_RMSE_acc.append(rmse)

    anRMSE = np.array(l_RMSE_acc)
    N = np.array(l_num)

    anMSE = np.square(anRMSE)

    anSE = anMSE * N

    mse = anSE.sum() / N.sum()

    rmse = np.sqrt(mse)

    print(f"RMSE = {rmse:.2f}")




lgbm(df)
flat_regression_by_county_0(df)
flat_regression_all(df)


    sns.scatterplot(x=n_basement, y= df['log_radon'].values)

    sns.lineplot(x = [0, 1], y = [lr.intercept_, lr.intercept_ + lr.coef_[0]])

    plt.show()