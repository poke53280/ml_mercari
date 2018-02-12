


import os ; os.environ['OMP_NUM_THREADS'] = '4'

import gc
import time
from time import gmtime, strftime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.linear_model import Ridge

import sys

import wordbatch

from wordbatch.models import FTRL, FM_FTRL

from wordbatch.extractors import WordBag, WordHash

from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from nltk.corpus import stopwords
import re

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250

###############################################################################################
#
#   is_stop
#
#

def is_stop():
    f = open(DATA_DIR + "stopfile.txt")
    s = f.read()
    f.close()

    isStop = s[:4] == 'stop'

    if (isStop):
        print("Processing stopped by stop file")

    return isStop

w = 90

def TXTP_bottom_digitize(x):
    if x < 3.5:
        return 3
    elif x < 4.5:
        return 4
    elif x < 5.5:
        return 5
    elif x < 6.5:
        return 6
    elif x < 7.5:
        return 7
    elif x < 8.5:
        return 8
    elif x < 9.5:
        return 9
    else:
        return x


w = 90
###############################################################################################
#
#   TXTP_rmsle
#

def TXTP_rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))

"""c"""

###############################################################################################
#
#   TXTP_split_cat
#

def TXTP_split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")

"ccc"

###############################################################################################
#
#   TXTP_handle_missing_inplace
#

def TXTP_handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)
    dataset['name'].fillna(value='missing', inplace=True)
    dataset['category_name'].fillna(value='missing', inplace=True)

w = 90

###############################################################################################
#
#   TXTP_cutting
#

def TXTP_cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'

###############################################################################################
#
#   TXTP_to_categorical
#

def TXTP_to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

w = 90

###############################################################################################
#
#   TXTP_normalize_text
#

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def TXTP_normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

w = 90


# ------------------------copy end


def load_train(isHome):

    isQuickRun = True

    if isHome:
        DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
        DATA_DIR_BASEMENT = "D:\\mercari\\"
        DATA_DIR = DATA_DIR_PORTABLE

        train = pd.read_table(DATA_DIR + "train.tsv");

    else:
        train = pd.read_table('../input/train.tsv', engine='c')

    train = train.drop(train[(train.price < 3.0)].index)

    return train

"""c"""

def get_X_name(df):
    df['name'] = df['name'].apply(lambda x: TXTP_normalize_text(x))

    wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None
                                                                  }), procs=8)
    wb.dictionary_freeze= True
  
    X_name = wb.fit_transform(df['name'])
    
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    return X_name

"""c"""

def get_category_Xs(df):
    wb = CountVectorizer()
    X_category1 = wb.fit_transform(df['general_cat'])
    X_category2 = wb.fit_transform(df['subcat_1'])
    X_category3 = wb.fit_transform(df['subcat_2'])

    return X_category1, X_category2, X_category3

"""c"""

def get_category_name_X(df):
    lb = LabelBinarizer(sparse_output=True)
    X_category_name = lb.fit_transform(df.category_name.cat.codes)

    return X_category_name

"""c"""


def get_description_X(df):
    df['item_description'] = df['item_description'].apply(lambda x: TXTP_normalize_text(x))

    wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None}) , procs=8)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(df['item_description'])
    
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    return X_description

"""c"""

def get_brand_X(df):
    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(df['brand_name'])

    return X_brand

def get_dummies_X(df):
    return csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)


"""c"""

def getX(df, isQuickRun):


    # np.count_nonzero(np.isnan(a))
    
    
    df['general_cat'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].apply(lambda x: TXTP_split_cat(x)))

    TXTP_handle_missing_inplace(df)

    TXTP_cutting(df)

    TXTP_to_categorical(df)

    l = []


    X_category_name = get_category_name_X(df)
    l.append(X_category_name)


    if isQuickRun:
        pass
    else:
        X_name = get_X_name(df)
        l.append(X_name)


    X_category1,X_category2, X_category3 =  get_category_Xs(df)
    l.append(X_category1)
    l.append(X_category2)
    l.append(X_category3)
    
    if isQuickRun:
        pass
    else:
        X_description = get_description_X(df)
        l.append(X_description)

    X_brand = get_brand_X(df)
    l.append(X_brand)
   
    X_dummies = get_dummies_X(df)
    l.append(X_dummies)

    X = hstack(l).tocsr()
  

    mask = np.array(np.clip(X.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X = X[:, mask]

    X = X.astype(np.float64)

    return X

"""c"""

def Huber_train(train_X, valid_X, train_y, valid_y, isQuickRun):

    setup_Huber = 1

    if isQuickRun:
        setupHuber = 4

    if (setup_Huber==1):
        model = HuberRegressor(fit_intercept=True, alpha=0.01, 
                               max_iter=80, epsilon=363)
    
    if (setup_Huber==2):
        model = HuberRegressor(fit_intercept=True, alpha=0.05, 
                               max_iter=200, epsilon=1.2)
                               
    if (setup_Huber==3):
        model = HuberRegressor(fit_intercept=True, alpha=0.02, 
                               max_iter=200, epsilon=256)  
        
    if (setup_Huber==4):
        model = HuberRegressor(fit_intercept=True, alpha=0.02, 
                               max_iter=2, tol = 0.01)

    model.fit(train_X, train_y)

    preds = model.predict(X=valid_X)

    print("Huber dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(preds)))

    return model.predict(X=train_X)

"""c"""

def FTRL_train(train_X, valid_X, train_y, valid_y, isQuickRun):

    if isQuickRun:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=train_X.shape[1], iters=3, inv_link="identity", threads=1)
    else:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=Xtrain_Xshape[1], iters=50, inv_link="identity", threads=1)
   
    model.fit(train_X, train_y)

    predsValid = model.predict(X=valid_X)

    print("FTRL dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(predsValid)))

    d = {}

    d['valid'] = predsValid
    d['train'] = model.predict(X=train_X)

    return d
   
"""c"""

def FM_FTRL_train(train_X, valid_X, train_y, valid_y, isQuickRun):

    if isQuickRun:
        model = HuberRegressor(fit_intercept=True, alpha=0.01, max_iter=8, epsilon=363)
    else:
        model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                                                                       D_fm=200, e_noise=0.0001, iters=16, inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    
    preds = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(preds)))

    d = {}

    d['valid'] = preds
    d['train'] = model.predict(X=train_X)

    return d

"""c"""

def PAR_train(train_X, valid_X, train_y, valid_y, isQuickRun): 

    setup_PAR = 2

    if isQuickRun:
        setup_PAR = 3
    
    if (setup_PAR==1):
        model = PassiveAggressiveRegressor(C=1.05, fit_intercept=True, loss='epsilon_insensitive', max_iter=120, random_state=433)
              
    if (setup_PAR==2):
        model = PassiveAggressiveRegressor(C=2.05, 
              fit_intercept=True, loss='epsilon_insensitive',
              max_iter=150, random_state=3232)     
        
    if (setup_PAR==3):
        model = PassiveAggressiveRegressor(C=2.05, 
              fit_intercept=True, loss='epsilon_insensitive',
              max_iter=2, random_state=3232) 
    
    model.fit(train_X, train_y)
    
    preds = model.predict(X=valid_X)
    print("PAR dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(preds)))

    d = {}

    d['valid'] = preds
    d['train'] = model.predict(X=train_X)

    return d

def LGB_train(train_X, valid_X, train_y, valid_y, isQuickRun):
    params = {
        'learning_rate': 0.6,
        'application': 'regression',
        'max_depth': 4,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.65,
        'nthread': 4,
        'min_data_in_leaf': 110,
        'max_bin': 40
    }

    # Remove features with document frequency <=100
    print(train_X.shape)

    mask = np.array(np.clip(train_X.getnnz(axis=0) - 100, 0, 1), dtype=bool)

    train_X_Trimmed = train_X[:, mask]
       
    print(train_X_Trimmed.shape)
    
    d_train = lgb.Dataset(train_X_Trimmed, label=train_y)
    del train_X_Trimmed
    gc.collect()
    
    d_valid = lgb.Dataset(valid_X, label=valid_y)
  
    watchlist = [d_train, d_valid]

    if isQuickRun:
        model = lgb.train(params, train_set=d_train, num_boost_round=180, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)
    else:
        model = lgb.train(params, train_set=d_train, num_boost_round=6800, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)

    preds = model.predict(valid_X)

    print("LGBM  RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(preds)))

    d = {}

    d['valid'] = preds
    d['train'] = model.predict(X=train_X)

    return d


"""c"""

def trainSingleSplit(train_X, valid_X, train_y, valid_y, isQuickRun):
    
    l = []

    l.append(FTRL_train(train_X, valid_X, train_y, valid_y, isQuickRun))
    gc.collect()

    #l.append(Huber_train(train_X, valid_X, train_y, valid_y, isQuickRun))
    #gc.collect()

    l.append(PAR_train(train_X, valid_X, train_y, valid_y, isQuickRun))
    gc.collect()

    l.append(FM_FTRL_train(train_X, valid_X, train_y, valid_y, isQuickRun))
    gc.collect()

    l.append(LGB_train(train_X, valid_X, train_y, valid_y, isQuickRun))
    gc.collect()


    X_s = hstack(l).tocsr()

    model = Ridge(alpha=10, max_iter=50000)

    return l[0] * 0.1 + l[1] * 0.2 + l[2] * 0.3 + l[3] * 0.4;

   
def trainCV(X, y, splits, isQuickRun):

    #train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=100)

    kf = KFold(n_splits = splits)
    
    nSplits = kf.get_n_splits(X)

    nFold = 0

    l_score = []

    y_stacked = np.zeros(len (y))

    for train_index, valid_index in kf.split(X):
        if is_stop():
            break

        print ("FOLD# " + str(nFold))

        train_X = X[train_index]  
        train_y = y[train_index]

        valid_X = X[valid_index]
        valid_y = y[valid_index]

        price_valid_real = np.expm1(valid_y)

        preds = trainSingleSplit(train_X, valid_X, train_y, valid_y, isQuickRun)

        y_stacked[valid_index] = preds

        price_pred = np.expm1(preds)
        o_lgbm = TXTP_rmsle(price_pred, price_valid_real)

        print ("RMSLE: " + str(o_lgbm))
        l_score.append(o_lgbm)

        nFold = nFold + 1

    w = 90
    a_lgbm = np.array(l_score)
    print ("STACK LGBM-RIDGE-HUBER (META: LGBM) RMSLE = " + str (a_lgbm.mean()) + " +/- " + str(a_lgbm.std()))

    return y_stacked

w = 90
   


def proc2():
    isHome = True
    isQuickRun = True

    df = load_train(isHome)

    y = np.log1p(df["price"])
    y = y.values

    X = getX(df, isQuickRun)

    del df
    gc.collect()


    DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
    DATA_DIR_BASEMENT = "D:\\mercari\\"
    DATA_DIR = DATA_DIR_PORTABLE

    y_pred = trainCV(X, y, 5, isQuickRun)




    


   
    
def process():
   
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    isHome = False

    isQuickRun = True

    if isHome:
        DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
        DATA_DIR_BASEMENT = "D:\\mercari\\"
        DATA_DIR = DATA_DIR_PORTABLE


        train = pd.read_table(DATA_DIR + "train.tsv");
        test = pd.read_table(DATA_DIR + "test.tsv");

    else:
        train = pd.read_table('../input/train.tsv', engine='c')
        test = pd.read_table('../input/test.tsv', engine='c')
  
    nrow_test = train.shape[0]  # -dftt.shape[0]
    dftt = train[(train.price < 3.0)]
    train = train.drop(train[(train.price < 3.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]

    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = zip(*merge['category_name'].apply(lambda x: TXTP_split_cat(x)))

    TXTP_handle_missing_inplace(merge)

    TXTP_cutting(merge)

    TXTP_to_categorical(merge)
    
    m = merge[:nrow_train].category_name.cat.codes
   
    _, cat3coded_valid = train_test_split(m, test_size=0.3, random_state=100)

    del m
    merge.drop('category_name', axis=1, inplace=True)

    lb = LabelBinarizer(sparse_output=True)

    X_cat3_valid = lb.fit_transform(cat3coded_valid)


    merge['name'] = merge['name'].apply(lambda x: TXTP_normalize_text(x))


    wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None
                                                                  }), procs=8)
    wb.dictionary_freeze= True
  
    X_name = wb.fit_transform(merge['name'])


    del(wb)
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['general_cat'])
    X_category2 = wb.fit_transform(merge['subcat_1'])
    X_category3 = wb.fit_transform(merge['subcat_2'])

    merge['item_description'] = merge['item_description'].apply(lambda x: TXTP_normalize_text(x))

    wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None}) , procs=8)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(merge['item_description'])
    
    del(wb)
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()

    del X_dummies, merge, X_description, lb, X_brand, X_category1, X_category2, X_category3, X_name; gc.collect()

    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=100)
    del X; gc.collect()

    # FTRL BEGIN

    if isQuickRun:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=1, inv_link="identity", threads=1)
    else:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=50, inv_link="identity", threads=1)
   

    #=> 0.438, 485s.

    model.fit(train_X, train_y)
    
    preds_FTRL = model.predict(X=valid_X)
    print("FTRL dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(preds_FTRL)))


    # FTRL END

    # HUBER BEGIN

    setup_Huber = 1

    if isQuickRun:
        setupHuber = 4

    if (setup_Huber==1):
        model = HuberRegressor(fit_intercept=True, alpha=0.01, 
                               max_iter=80, epsilon=363)
    
    if (setup_Huber==2):
        model = HuberRegressor(fit_intercept=True, alpha=0.05, 
                               max_iter=200, epsilon=1.2)
                               
    if (setup_Huber==3):
        model = HuberRegressor(fit_intercept=True, alpha=0.02, 
                               max_iter=200, epsilon=256)  
        
    if (setup_Huber==4):
        model = HuberRegressor(tol = 0.02)
                        

    # => 0.476, 587 s.

    model.fit(train_X, train_y)
    predsHUBER = model.predict(X=valid_X)

    print("HUBER  RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(predsHUBER)))

    # HUBER END

    # PASSIVE AGRESSIVE BEGIN
    setup_PAR = 2

    if isQuickRun:
        setup_PAR = 3
    
    if (setup_PAR==1):
        model = PassiveAggressiveRegressor(C=1.05, fit_intercept=True, loss='epsilon_insensitive', max_iter=120, random_state=433)
              
    if (setup_PAR==2):
        model = PassiveAggressiveRegressor(C=2.05, 
              fit_intercept=True, loss='epsilon_insensitive',
              max_iter=150, random_state=3232)     
        
    if (setup_PAR==3):
        model = PassiveAggressiveRegressor(C=2.05, 
              fit_intercept=True, loss='epsilon_insensitive',
              max_iter=2, random_state=3232) 
    

    # => 0.599, 229s.
    
    model.fit(train_X, train_y)
    predsPAR = model.predict(X=valid_X)

    print("PASSIVE AGRESSIVE RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(predsPAR)))

    # PASSIVE AGRESSIVE END


    # FM_FTRL BEGIN

    if isQuickRun:
        model = HuberRegressor(fit_intercept=True, alpha=0.01, max_iter=8, epsilon=363)
    else:
        model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                                                                       D_fm=200, e_noise=0.0001, iters=16, inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    del train_X, train_y; gc.collect()
    
    predsFM_FTRL = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(predsFM_FTRL)))

   

    # FM_FTRL END

    del X_test; gc.collect()
    params = {
        'learning_rate': 0.6,
        'application': 'regression',
        'max_depth': 4,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.65,
        'nthread': 4,
        'min_data_in_leaf': 110,
        'max_bin': 40
    }

    # Remove features with document frequency <=100
    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print(sparse_merge.shape)
    del sparse_merge; gc.collect()
    
    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=100)
    del X, y; gc.collect()
    d_train = lgb.Dataset(train_X, label=train_y)
    del train_X, train_y; gc.collect()
    
    d_valid = lgb.Dataset(valid_X, label=valid_y)
  
    watchlist = [d_train, d_valid]

    if isQuickRun:
        model = lgb.train(params, train_set=d_train, num_boost_round=180, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)
    else:
        model = lgb.train(params, train_set=d_train, num_boost_round=6800, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)

    del d_train; gc.collect()

    predsLGB = model.predict(valid_X)
    del valid_X; gc.collect()

    print("LGB dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(predsLGB)))
          
    d = X_cat3_valid.todense()

    X_s = csr_matrix(np.column_stack((preds_FTRL, predsFM_FTRL, predsLGB, predsHUBER, predsPAR, d )))
   
    
    # Stacking
    
    d_train = lgb.Dataset(X_s, label=valid_y)
    watchlist = [d_train]

    params = {
        'learning_rate': 0.001,
        'application': 'regression',
        'max_depth': 4,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.65,
        'nthread': 4,
        'min_data_in_leaf': 110,
        'max_bin': 8191
    }

    model = lgb.train(params, train_set=d_train, num_boost_round=9800, valid_sets=watchlist, verbose_eval=1000)

    preds = model.predict(X_test)

    submission['price'] =  np.expm1(preds)                  
    submission.to_csv("submission_wordbatch_ftrl_fm_lgb.csv", index=False)

    print("All done.")


w = 90

if __name__ == '__main__':
    process()

