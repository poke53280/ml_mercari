


import os ; os.environ['OMP_NUM_THREADS'] = '4'

import psutil
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

from scipy import sparse

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250

"""c"""

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

def create_wordBatchForName():
    wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None
                                                                  }), procs=8)
    wb.dictionary_freeze= True

    return wb


def get_X_name_train(df, wb):
    df['name'] = df['name'].apply(lambda x: TXTP_normalize_text(x))
  
    X_name = wb.fit_transform(df['name'])
    
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    return X_name

"""c"""

def get_X_name_test(df, wb):
    df['name'] = df['name'].apply(lambda x: TXTP_normalize_text(x))
  
    X_name = wb.transform(df['name'])
    
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

def create_wb_for_description():
    wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None}) , procs=8)
    wb.dictionary_freeze= True

    return wb

def get_description_X_train(df, wb):
    df['item_description'] = df['item_description'].apply(lambda x: TXTP_normalize_text(x))
    
    X_description = wb.fit_transform(df['item_description'])
    
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]

    return X_description

"""c"""

def get_description_X_test(df, wb):
    df['item_description'] = df['item_description'].apply(lambda x: TXTP_normalize_text(x))
    
    X_description = wb.transform(df['item_description'])
    
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

def getXTrain(df, isQuickRun):


    train_bits = {}

    
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
        wb = create_wordBatchForName()
        X_name = get_X_name_train(df, wb)

        train_bits['name_encoder'] = wb

        l.append(X_name)


    X_category1,X_category2, X_category3 = get_category_Xs(df)
    l.append(X_category1)
    l.append(X_category2)
    l.append(X_category3)
    
    if isQuickRun:
        pass
    else:
        wb2 = create_wb_for_description()
        X_description = get_description_X_train(df,wb2)
        l.append(X_description)
        train_bits['desc_encoder'] = wb2

    X_brand = get_brand_X(df)
    l.append(X_brand)
   
    X_dummies = get_dummies_X(df)
    l.append(X_dummies)

    X = hstack(l).tocsr()
  

    mask = np.array(np.clip(X.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X = X[:, mask]

    X = X.astype(np.float64)

    d = {}
    d['X'] = X
    d['bits'] = train_bits

    return d

"""c"""
def getXTest(df, isQuickRun, train_bits):

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
        wb = train_bits['name_encoder']
        X_name = get_X_name_test(df, wb)
        l.append(X_name)


    X_category1,X_category2, X_category3 = get_category_Xs(df)
    l.append(X_category1)
    l.append(X_category2)
    l.append(X_category3)
    
    if isQuickRun:
        pass
    else:
        wb = train_bits['desc_encoder']
        X_description = get_description_X_test(df, wb)
        l.append(X_name)

    X_brand = get_brand_X(df)
    l.append(X_brand)
   
    X_dummies = get_dummies_X(df)
    l.append(X_dummies)

    X = hstack(l).tocsr()
  

    mask = np.array(np.clip(X.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X = X[:, mask]

    X = X.astype(np.float64)

    return X



def FTRL_train(train_X, train_y, isQuickRun):

    if isQuickRun:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=train_X.shape[1], iters=3, inv_link="identity", threads=4)
    else:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=train_X.shape[1], iters=47, inv_link="identity", threads=4)
   
    model.fit(train_X, train_y)

    return model
   
"""c"""

def FM_FTRL_train(train_X, train_y, isQuickRun):

    if isQuickRun:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=train_X.shape[1], iters=7, inv_link="identity", threads=4)
    else:
        model = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=train_X.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=18, inv_link="identity", threads=4)

    model.fit(train_X, train_y)

    return model

"""c"""

def LGB_train(train_X, train_y, isQuickRun):
    params = {
        'learning_rate': 0.57,
        'application': 'regression',
        'max_depth': 5,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.65,
        'nthread': 4,
        'min_data_in_leaf': 100,
        'max_bin': 31
    }

    # Remove features with document frequency <=100
    #print(train_X.shape)

    mask = np.array(np.clip(train_X.getnnz(axis=0) - 100, 0, 1), dtype=bool)

    train_X_Trimmed = train_X[:, mask]
       
    #print(train_X_Trimmed.shape)
    
    d_train = lgb.Dataset(train_X_Trimmed, label=train_y)
    del train_X_Trimmed
    gc.collect()
    
    watchlist = [d_train]

    if isQuickRun:
        model = lgb.train(params, train_set=d_train, num_boost_round=100, valid_sets=watchlist, verbose_eval=0)
    else:
        model = lgb.train(params, train_set=d_train, num_boost_round=4800, valid_sets=watchlist, verbose_eval=0)

    return model


"""c"""

def trainStack(X_s, y, splits, isQuickRun):
    kf = KFold(n_splits = splits)
    
    nSplits = kf.get_n_splits(X_s)

    nFold = 0

    y_pred = np.zeros(len (y))

    for train_index, valid_index in kf.split(X_s):

        print ("FOLD# " + str(nFold))

        train_X = X_s[train_index]  
        train_y = y[train_index]

        valid_X = X_s[valid_index]
        valid_y = y[valid_index]

        model = Ridge(alpha=10, max_iter=50000)
        
        model.fit(train_X, train_y)

        predsValid = model.predict(valid_X)

        print("stack RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(predsValid)))

        y_pred[valid_index] = predsValid

        nFold = nFold + 1

"""c"""

def trainAllModels(start_time, X, y, isQuickRun):
    lm = []

    print ("FTRL...")
    m_FTRL = FTRL_train(X, y, isQuickRun)
    lm.append(m_FTRL)

    gc.collect()
    print (psutil.virtual_memory().percent)
    print('[{}] Done FTRL'.format(time.time() - start_time))

    print ("FM...")
    m_FM = FM_FTRL_train(X, y, isQuickRun)
    lm.append(m_FM)

    gc.collect()
    print (psutil.virtual_memory().percent)
    print('[{}] Done FM'.format(time.time() - start_time))

    print ("LGBM...")
    m_LGB = LGB_train(X, y, isQuickRun)
    lm.append(m_LGB)

    gc.collect()
    print (psutil.virtual_memory().percent)
    print('[{}] Done LGBM'.format(time.time() - start_time))


    print("Done base models.")
    return lm

"""c"""

def predictOnline(lm, X_t):

    y_pred = []

    y_pred.append(lm[0].predict(X_t))
    y_pred.append(lm[1].predict(X_t))
    y_pred.append(lm[2].predict(X_t))

    X_s = sparse.csr_matrix(np.column_stack((y_pred[0], y_pred[1], y_pred[2])))

    # meta regressor

    y = lm[3].predict(X_s)

    return y


"""c"""

def trainOnline(start_time, X, y, isQuickRun):

    y_pred = []

    lm = trainAllModels(start_time, X, y, isQuickRun)

    gc.collect()
    print (psutil.virtual_memory().percent)
    print('[{}] Done base level training'.format(time.time() - start_time))

    for m in lm:
        p = m.predict(X)
        y_pred.append(p)

    X_s = sparse.csr_matrix(np.column_stack((y_pred[0], y_pred[1], y_pred[2])))

    model = Ridge(alpha=10, max_iter=50000)

    print('[{}] Ridge meta processing'.format(time.time() - start_time))
        
    model.fit(X_s, y)

    print('[{}] Ridge meta completed'.format(time.time() - start_time))

    lm.append(model)

    return lm
   
def trainCV(X, y, splits, isQuickRun):

    #train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=100)

    kf = KFold(n_splits = splits)
    
    nSplits = kf.get_n_splits(X)

    nFold = 0

    y_pred = []

    y_pred.append(np.zeros( len (y)))
    y_pred.append(np.zeros( len (y)))
    y_pred.append(np.zeros( len (y)))


    for train_index, valid_index in kf.split(X):

        print ("FOLD# " + str(nFold))

        train_X = X[train_index]  
        train_y = y[train_index]

        lm = trainAllModels(start_time, train_X, train_y, isQuickRun)

        valid_X = X[valid_index]
        valid_y = y[valid_index]

        y_col = 0

        for m in lm:
             p = m.predict(valid_X)
             print("  RMSLE mod #" + str(y_col) + " " +str(TXTP_rmsle(np.expm1(valid_y), np.expm1(p))))

             y_pred[y_col][valid_index] = p
             y_col = y_col + 1

        nFold = nFold + 1
        
    X_s = sparse.csr_matrix(np.column_stack((y_pred[0], y_pred[1], y_pred[2])))

    return X_s
  

"""c"""
  

  
def deliver_run(start_time, isHome, isQuickTrain, isQuickPreprocess):
    # Prepare for prediction

    print (psutil.virtual_memory().percent)
    print('[{}] Loading train data...'.format(time.time() - start_time))
    
    df = load_train(isHome)

    print (psutil.virtual_memory().percent)
    print('[{}] Preprocessing...'.format(time.time() - start_time))

    y = np.log1p(df["price"])
    y = y.values

    d = getXTrain(df, isQuickPreprocess)

    X = d['X']

    train_bits = d['bits']
   

    del df
    gc.collect()

    print (psutil.virtual_memory().percent)
    print('[{}] Training...'.format(time.time() - start_time))
   
    lm = trainOnline(start_time, X, y, isQuickTrain)

    print ("Training done.")

    del X
    del y
    gc.collect()

    print (psutil.virtual_memory().percent)

    print('[{}] Test prediction in chunks...'.format(time.time() - start_time))

    CHUNK_SIZE = 100 *1000

    if isHome:
        DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
        DATA_DIR_BASEMENT = "D:\\mercari\\"
        DATA_DIR = DATA_DIR_PORTABLE

        reader = pd.read_table(DATA_DIR + "test.tsv", chunksize=CHUNK_SIZE)

    else:
        reader = pd.read_table('../input/test.tsv', chunksize=CHUNK_SIZE, engine='c')

    y_out = []

    test_id_out = []


    for df_t in reader:
        print("Prediciting " + str(len(df_t)) + " items...")
        print('[{}] Test prediction in chunks...'.format(time.time() - start_time))
        print (psutil.virtual_memory().percent)

        X_t = getXTest(df_t, isQuickPreprocess, train_bits)
        y_p = predictOnline(lm, X_t)
        y_out.extend(y_p)

        test_id_out.extend(df_t.test_id.values)

    data_tuples = list(zip(test_id_out,np.expm1(y_out)))

    submission = pd.DataFrame(data_tuples, columns=['test_id','price'])

    if isHome:
        submission.to_csv(DATA_DIR + "submission_anders.csv", index=False)
    else:
        submission.to_csv("submission_anders.csv", index=False)

    print("Stored test lines: " + str(len (submission)))
    print('[{}] All done.'.format(time.time() - start_time))



"""c"""

def dev_run(isHome, isQuickRun):

    print (psutil.virtual_memory().percent)

    df = load_train(isHome)

    print (psutil.virtual_memory().percent)

    y = np.log1p(df["price"])
    y = y.values

    X = getX(df, isQuickRun)

    del df
    gc.collect()

    print (psutil.virtual_memory().percent)

    X_s = trainCV(X, y, 5, isQuickRun)

    print (psutil.virtual_memory().percent)

    trainStack(X_s, y, 5, isQuickRun)

    print (psutil.virtual_memory().percent)

if __name__ == '__main__':
    start_time = time.time()
    isHome = True
    isQuickTrain = True
    isQuickPreprocess = False
    deliver_run(start_time, isHome, isQuickTrain, isQuickPreprocess) 


