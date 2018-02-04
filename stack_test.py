


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

import sys

import wordbatch

from wordbatch.models import FTRL, FM_FTRL

from wordbatch.extractors import WordBag, WordHash

from sklearn.linear_model import HuberRegressor



from nltk.corpus import stopwords
import re

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250

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


###############################################################################################
#
#   TXTP_split_cat
#

def TXTP_split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


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



def process():
    start_time = time.time()
   
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    isHome = True

    if isHome:
        DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
        DATA_DIR_BASEMENT = "D:\\mercari\\"
        DATA_DIR = DATA_DIR_PORTABLE


        train = pd.read_table(DATA_DIR + "train.tsv");
        test = pd.read_table(DATA_DIR + "test.tsv");

    else:
        train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
        test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')
  
    nrow_test = train.shape[0]  # -dftt.shape[0]
    dftt = train[(train.price < 3.0)]
    train = train.drop(train[(train.price < 3.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]

    #investigate_price(train["price"])

    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: TXTP_split_cat(x)))
    
    
    

    TXTP_handle_missing_inplace(merge)

    TXTP_cutting(merge)

    TXTP_to_categorical(merge)
    
    merge['coded'] = merge[:nrow_train].category_name.cat.codes
   
    _, cat3coded_valid = train_test_split(merge['coded'], test_size=0.3, random_state=100)

    merge.drop('category_name', axis=1, inplace=True)

    lb = LabelBinarizer(sparse_output=True)

    X_cat3_valid = lb.fit_transform(cat3coded_valid)


    merge['name'] = merge['name'].apply(lambda x: TXTP_normalize_text(x))

    print('[{}] Name normalization completed'.format(time.time() - start_time))

    wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None
                                                                  }), procs=8)
    wb.dictionary_freeze= True
  


    X_name = wb.fit_transform(merge['name'])

    print('[{}] name fit_transform completed'.format(time.time() - start_time))

    del(wb)
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['general_cat'])
    X_category2 = wb.fit_transform(merge['subcat_1'])
    X_category3 = wb.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))


    merge['item_description'] = merge['item_description'].apply(lambda x: TXTP_normalize_text(x))

    print('[{}] item_description normalization completed.'.format(time.time() - start_time))

    wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None})
                             , procs=8)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(merge['item_description'])
    
    print('[{}] item_description fit_transform completed'.format(time.time() - start_time))

    del(wb)
    X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape)
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()

    print('[{}] Create sparse merge completed'.format(time.time() - start_time))
    del X_dummies, merge, X_description, lb, X_brand, X_category1, X_category2, X_category3, X_name; gc.collect()

    # pd.to_pickle((sparse_merge, y), DATA_DIR + "xy_anders.pkl")
    # else:
    #nrow_train, nrow_test= 1481661, 1482535
    #sparse_merge, y = pd.read_pickle("xy.pkl")

    # Remove features with document frequency <=1
    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print(sparse_merge.shape)
    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=100)

    if isHome:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=1, inv_link="identity", threads=1)
    else:
        model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=50, inv_link="identity", threads=1)

    del X; gc.collect()
    model.fit(train_X, train_y)
    print('[{}] Train FTRL completed'.format(time.time() - start_time))

    
    preds_FTRL = model.predict(X=valid_X)
    print("FTRL dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(preds_FTRL)))

    print('[{}] Predict FTRL completed'.format(time.time() - start_time))

    if isHome:
        model = HuberRegressor(fit_intercept=True, alpha=0.01, max_iter=8, epsilon=363)
    else:
        model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                                                                       D_fm=200, e_noise=0.0001, iters=16, inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    del train_X, train_y; gc.collect()
    print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))
    
    predsFM_FTRL = model.predict(X=valid_X)
    print("FM_FTRL dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(predsFM_FTRL)))

   
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
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

    if isHome:
        model = lgb.train(params, train_set=d_train, num_boost_round=180, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)
    else:
        model = lgb.train(params, train_set=d_train, num_boost_round=6800, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)

    del d_train; gc.collect()

    predsLGB = model.predict(valid_X)
    del valid_X; gc.collect()

    print("LGB dev RMSLE:", TXTP_rmsle(np.expm1(valid_y), np.expm1(predsLGB)))
          
    del X_test; gc.collect()
    print('[{}] Predict LGB completed.'.format(time.time() - start_time))


    X_s = csr_matrix(np.column_stack((preds_FTRL, predsFM_FTRL, predsLGB)))
    
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

def main():
    process()



if __name__ == '__main__':
    main()

