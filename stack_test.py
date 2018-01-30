

# Fork: https://www.kaggle.com/serigne/wordbatch-ftrl-fm-lgb-lb-0-42441?scriptVersionId=2140580
# Based on Bojan -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# and Nishant -> https://www.kaggle.com/nishkgp/more-improved-ridge-2-lgbm

import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from sklearn.linear_model import HuberRegressor
from sklearn.feature_extraction.text import CountVectorizer

import sys

#  lasso (least absolute shrinkage and selection operator)
from sklearn.linear_model import Lasso

import time
start_time = time.time()
tcurrent   = start_time

np.random.seed(55)   

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
#sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re

NUM_BRANDS = 4550
NUM_CATEGORIES = 1280

#develop = False
develop= True

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

w = 90


def main():
    start_time = time.time()
    from time import gmtime, strftime
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # if 1 == 1:
    DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
    DATA_DIR_BASEMENT = "D:\\mercari\\"
    DATA_DIR = DATA_DIR_PORTABLE

    # df = pd.read_table(DATA_DIR + "train.tsv");


    #train = pd.read_table('../input/mercari-price-suggestion-challenge/train.tsv', engine='c')
    #test = pd.read_table('../input/mercari-price-suggestion-challenge/test.tsv', engine='c')

    train = pd.read_table(DATA_DIR + "train.tsv", engine='c')
    test = pd.read_table(DATA_DIR + "test.tsv", engine='c')

    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]  # -dftt.shape[0]
    dftt = train[(train.price < 3.0)]
    train = train.drop(train[(train.price < 3.0)].index)
    del dftt['price']
    nrow_train = train.shape[0]
    # print(nrow_train, nrow_test)
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, dftt, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))



    #wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0], "hash_size": 2 ** 29, "norm": None, "tf": 'binary', "idf": None}), procs=8)
    #wb.dictionary_freeze= True
    #X_name = wb.fit_transform(merge['name'])

    cv0 = CountVectorizer(min_df = 3)
    X_name = cv0.fit_transform(merge['name'])


    del(wb)
    X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['general_cat'])
    X_category2 = wb.fit_transform(merge['subcat_1'])
    X_category3 = wb.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    # wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None})
                             , procs=8)
    wb.dictionary_freeze= True
    X_description = wb.fit_transform(merge['item_description'])
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

    pd.to_pickle((sparse_merge, y), "xy.pkl")
    # else:
    #    nrow_train, nrow_test= 1481661, 1482535
    #    sparse_merge, y = pd.read_pickle("xy.pkl")

    # Remove features with document frequency <=1
    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print(sparse_merge.shape)
    train_X, train_y = X, y
    
    X_BACKUP = X
    y_BACKUP = y

    IN: train_X, train_y

    OUT:  
        predsF = model.predict(X_test)
        predsFM = model.predict(X_test)
        predsL = model.predict(X_test)
        predsH = model.predict(X=X_test)

    Create X_s as predsF, predsFM, predsL, predsH = train_y





    preds = predsH * 0.04 + predsF * 0.09 + predsL * 0.33 + predsFM * 0.54

    submission['price'] = np.expm1(preds)




    model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=30, inv_link="identity", threads=1)
    del X; gc.collect()
    model.fit(train_X, train_y)
    print('[{}] Train FTRL completed'.format(time.time() - start_time))

    predsF = model.predict(X_test)
    print('[{}] Predict FTRL completed'.format(time.time() - start_time))

    model = FM_FTRL(alpha=0.012, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], 
                    alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)

    model.fit(train_X, train_y)
    del train_X, train_y; gc.collect()
    print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))

    predsFM = model.predict(X_test)
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    del X_test; gc.collect()
    params = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 4,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.71,
        'bagging_freq': 5,
        'feature_fraction': 0.67,
        'nthread': 4,
        'min_data_in_leaf': 120,
        'max_bin': 40
    }

    # Remove features with document frequency <=200
    print(sparse_merge.shape)
    mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 200, 0, 1), dtype=bool)
    sparse_merge = sparse_merge[:, mask]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print(sparse_merge.shape)
    del sparse_merge; gc.collect()
    train_X, train_y = X, y
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)
    del X, y; gc.collect()
    d_train = lgb.Dataset(train_X, label=train_y)
   # del train_X, train_y; gc.collect()
    watchlist = [d_train]
    if develop:
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        del valid_y; gc.collect()
        watchlist = [d_train, d_valid]

    #model = lgb.train(params, train_set=d_train, num_boost_round=7500, valid_sets=watchlist, \
    #                  early_stopping_rounds=1000, verbose_eval=1000)
                      
    model = lgb.train(params, train_set=d_train, num_boost_round=6700, valid_sets=watchlist, \
                      early_stopping_rounds=1000, verbose_eval=1000)
                          
    del d_train; gc.collect()
    

    predsL = model.predict(X_test)
   # del X_test; gc.collect()
    print('[{}] Predict LGB completed.'.format(time.time() - start_time))
    
    #--- BEGIN Huber
    # Details: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
    model = HuberRegressor(fit_intercept=True, alpha=0.01, max_iter=80, epsilon=363)
    model.fit(train_X, train_y)
    print('[{}] Predict Huber completed.'.format(time.time() - start_time))
    predsH = model.predict(X=X_test)
    #--- END Huber

    # original
    # preds = (predsF * 0.2 + predsL * 0.3 + predsFM * 0.5)

    # modified 

    # brand cateogory. Length desc. Bundle prob, number of numbers. Guess on quantity.



    out :
        preds = y_s

    preds = predsH * 0.04 + predsF * 0.09 + predsL * 0.33 + predsFM * 0.54

    submission['price'] = np.expm1(preds)
    submission.to_csv("sub ftrl_fm_lgb_huber v2.csv", index=False)

    nm=(time.time() - start_time)/60
    print ("Total processing time %s min" % nm)

if __name__ == '__main__':
    main()

