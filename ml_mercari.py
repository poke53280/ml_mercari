
import pandas as pd
import pyximport; pyximport.install()
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score

import scipy

import time
import datetime

import gc


"""  
Text processing read:
    http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/
"""

"""Add to below: Find and return a category name with close to desired elements."""

"""Todo: Work in detail on a small category. Post improved kernel."""


"""--------------------------------------------------------------------------------------"""
# Based on Bojan's -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# Changes:
# 1. Split category_name into sub-categories
# 2. Parallelize LGBM to 4 cores
# 3. Increase the number of rounds in 1st LGBM
# 4. Another LGBM with different seed for model and training split, slightly different hyper-parametes.
# 5. Weights on ensemble
# 6. SGDRegressor doesn't improve the result, going with only 1 Ridge and 2 LGBM

import pyximport; pyximport.install()
import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb



def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):

    NUM_BRANDS = 4000
    NUM_CATEGORIES = 1000

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


def rmsle_func(y, y0):
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


def train_ensemble(train, test):
   
    NAME_MIN_DF = 10
    MAX_FEATURES_ITEM_DESCRIPTION = 50000

    start_time = time.time()
   
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    submission: pd.DataFrame = test[['test_id']]

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(merge['name'])
    print('[{}] Count vectorize `name` completed.'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_category1 = cv.fit_transform(merge['general_cat'])
    X_category2 = cv.fit_transform(merge['subcat_1'])
    X_category3 = cv.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    print('[{}] TFIDF vectorize `item_description` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    
    model = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=101, solver='auto', tol=0.001)
    model.fit(X, y)
    print('[{}] Train ridge completed'.format(time.time() - start_time))
    predsR = model.predict(X=X_test)
    print('[{}] Predict ridge completed'.format(time.time() - start_time))

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.15, random_state = 144) 
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
    watchlist = [d_train, d_valid]
    
    params = {
        'learning_rate': 0.4,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 80,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4,
        'max_bin': 255
    }

    params2 = {
        'learning_rate': 1,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 140,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 4,
        'max_bin': 255
    }

    model = lgb.train(params, train_set=d_train, num_boost_round=3000, valid_sets=watchlist, \
    early_stopping_rounds=250, verbose_eval=500) 
    predsL = model.predict(X_test)
    
    print('[{}] Predict lgb 1 completed.'.format(time.time() - start_time))
    
    train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101) 
    d_train2 = lgb.Dataset(train_X2, label=train_y2)
    d_valid2 = lgb.Dataset(valid_X2, label=valid_y2)
    watchlist2 = [d_train2, d_valid2]

    model = lgb.train(params2, train_set=d_train2, num_boost_round=3000, valid_sets=watchlist2, \
    early_stopping_rounds=250, verbose_eval=500) 
    predsL2 = model.predict(X_test)

    print('[{}] Predict lgb 2 completed.'.format(time.time() - start_time))

    preds = predsR*0.35 + predsL*0.35 + predsL2*0.3

    y_test = np.expm1(preds)

    price_series = pd.Series(y_test)
    index_series = pd.Series(test.test_id)

    index_series = index_series.reset_index()

    df = pd.concat([index_series, price_series], axis=1)
 
    df = df.drop(['test_id'], axis = 1)
   
    df.columns = ['test_id', 'price']

    return df


i = 90

def train_all(train, test):

    df = pd.concat([train, test], 0)
    nrow_train = train.shape[0]
    y_train = np.log1p(train["price"])

    NUM_BRANDS = 2500
    NAME_MIN_DF = 10
    MAX_FEAT_DESCP = 50000

    df["category_name"] = df["category_name"].fillna("Other").astype("category")
    df["brand_name"] = df["brand_name"].fillna("unknown")

    pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
    df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

    df["item_description"] = df["item_description"].fillna("None")
    df["item_condition_id"] = df["item_condition_id"].astype("category")
    df["brand_name"] = df["brand_name"].astype("category")

    print(df.memory_usage(deep = True))

    print("Encodings")
    count = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = count.fit_transform(df["name"])

    print("Category Encoders")
    unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
    count_category = CountVectorizer()
    X_category = count_category.fit_transform(df["category_name"])

    print("Descp encoders")

    count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                              ngram_range = (1,3),
                              stop_words = "english")
    X_descp = count_descp.fit_transform(df["item_description"])

    print("Brand encoders")

    vect_brand = LabelBinarizer(sparse_output=True)
    
    X_brand = vect_brand.fit_transform(df["brand_name"])

    print("Dummy Encoders")

    X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[["item_condition_id", "shipping"]], sparse = True).values)

    X = scipy.sparse.hstack((X_dummies, X_descp, X_brand, X_category, X_name)).tocsr()

    print([X_dummies.shape, X_category.shape, X_name.shape, X_descp.shape, X_brand.shape])

    X_train = X[:nrow_train]
    model = Ridge(solver = "lsqr", fit_intercept=False)

    print("Fitting Model")
    model.fit(X_train, y_train)

    X_test = X[nrow_train:]
    y_test = model.predict(X_test)


    y_test = np.expm1(y_test)

    price_series = pd.Series(y_test)
    index_series = pd.Series(test.test_id)

    index_series = index_series.reset_index()

    df = pd.concat([index_series, price_series], axis=1)
 
    df = df.drop(['test_id'], axis = 1)
   
    df.columns = ['test_id', 'price']

    return df


w = 99


def train_single_category(train, test, isGBM):
   
    nrow_train = train.shape[0]

    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])

    merge['brand_name'].fillna(value='missing', inplace=True)
    merge['item_description'].fillna(value='missing', inplace=True)
    
    cv0 = TfidfVectorizer(ngram_range=(1,3))

    X_name = cv0.fit_transform(merge['name'])

    lb0 = LabelBinarizer(sparse_output=True)
   
    tv = TfidfVectorizer(ngram_range=(1, 5))

    X_description = tv.fit_transform(merge['item_description'])

    lb = LabelBinarizer(sparse_output=True)

    merge['brand_name'].fillna(value='missing', inplace=True)
    
    X_brand = lb.fit_transform(merge['brand_name'])

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_name)).tocsr()

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.31, random_state = 144)

    if isGBM:
        print("GBM...")
        d_train = lgb.Dataset(train_X, label=train_y)
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        
        watchlist = [d_train, d_valid]
    
        params = {
            'learning_rate': 0.4,
            'application': 'regression',
            'max_depth': 3,
            'num_leaves': 80,
            'verbosity': -1,
            'metric': 'RMSE',
            'data_random_seed': 1,
            'bagging_fraction': 0.5,
            'nthread': 4,
            'max_bin': 255
        }
   
        model = lgb.train(params, train_set=d_train, num_boost_round=3000, valid_sets=watchlist, early_stopping_rounds=250, verbose_eval=500) 
        valid_y_pred = model.predict(valid_X)
        valid_y_pred = np.expm1(valid_y_pred)
        valid_y =  np.expm1(valid_y)
        o = rmsle_func(valid_y_pred, valid_y)

    else:
        print("RIDGE...")
        model = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=100, normalize=False, random_state=101, solver='auto', tol=0.001)
        model.fit(train_X, train_y)
        valid_y_pred = model.predict(valid_X)
        valid_y_pred = np.expm1(valid_y_pred)
        valid_y =  np.expm1(valid_y)
        o = rmsle_func(valid_y_pred, valid_y)
        model.fit(X, y)

    y_test = model.predict(X_test)

    y_test = np.expm1(y_test)

    price_series = pd.Series(y_test)
    index_series = pd.Series(test.test_id)

    index_series = index_series.reset_index()

    df = pd.concat([index_series, price_series], axis=1)
 
    df = df.drop(['test_id'], axis = 1)
   
    df.columns = ['test_id', 'price']

    dict = {'df':df, 'rmsle':o }

    return dict


i = 323
  
def main():

    start_time = time.time()

    print('[{}] Go'.format(time.time() - start_time))

    isOnline = False
    isReverse = False

    if (isOnline):
        full_train = pd.read_table('../input/train.tsv', engine='c')
        full_test = pd.read_table('../input/test.tsv', engine='c')
    else:
        DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
        DATA_DIR_BASEMENT = "D:\\mercari\\"
        DATA_DIR = DATA_DIR_PORTABLE
        full_train = pd.read_table(DATA_DIR + "train.tsv");
        full_test = pd.read_csv(DATA_DIR + "test.tsv", sep = "\t", encoding="utf-8", engine = "python");



    """Todo: Remove price 0 items"""

    p = full_train.category_name.value_counts()

    dfCat = pd.DataFrame(p)

    if isReverse:
        dfCat = dfCat.reindex(index=dfCat.index[::-1])

    dfCat = dfCat.reset_index()

    dfCat.columns = ['name', 'counter']

    cat_max = 400

    cat_counter = 0

    df_acc = pd.DataFrame()

    train_acc = 0
    test_acc = 0

    processed_cats = []
   
    
    while cat_counter < cat_max:
        single_cat = dfCat.name[cat_counter]

        cat_counter = cat_counter + 1

        test = full_test.loc[full_test.category_name == single_cat]
        train = full_train.loc[full_train.category_name == single_cat]

        item_num_string = " " + str(len(train)) + "/" + str(len(test)) + " item(s)"

        isTestElementsPresent = (len(test) > 0)
        
        if isTestElementsPresent:
           
            print("Start GBM")
            dict_GBM = train_single_category(train, test, True)
            df_GBM = dict_GBM['df']
            rmsle_GBM = dict_GBM['rmsle']

            print("Start RIDGE")
            dict_RIDGE = train_single_category(train, test, False)
            df_RIDGE = dict_RIDGE['df']
            rmsle_RIDGE = dict_RIDGE['rmsle']

            if (rmsle_GBM < 0.42) | (rmsle_RIDGE < 0.42):
                test_acc = test_acc + len(test)
                train_acc = train_acc + len (train)

                if rmsle_GBM < rmsle_RIDGE:
                    print('[{0:4.0f}] KEEP LGBM  R {1:3.3f} G {2:3.3f} '.format(time.time() - start_time, rmsle_RIDGE, rmsle_GBM) + single_cat + item_num_string)
                    df_acc = pd.concat([df_GBM, df_acc], axis = 0)
                else:
                    print('[{0:4.0f}] KEEP RIDGE R {1:3.3f} G {2:3.3f} '.format(time.time() - start_time, rmsle_RIDGE, rmsle_GBM) + single_cat + item_num_string)
                    df_acc = pd.concat([df_RIDGE, df_acc], axis = 0)
                
                processed_cats.append(single_cat)
            else:
                print('[{0:4.0f}] DROP R {1:3.3f} G {2:3.3f} '.format(time.time() - start_time, rmsle_RIDGE, rmsle_GBM) + single_cat + item_num_string)


        col = gc.collect()

    v = 90


    print('[{}] Begin processing of remaining data'.format(time.time() - start_time))

    full_train.category_name.fillna("missing", inplace = True)
    full_test.category_name.fillna("missing", inplace = True)

    train2 = full_train[~full_train.category_name.isin(processed_cats)]
    test2 =  full_test[~full_test.category_name.isin(processed_cats)]

    df = train_ensemble(train2, test2)
    df_acc = pd.concat([df, df_acc], axis = 0)
    
    test_acc = test_acc + len(test2)
    train_acc = train_acc + len (train2)

    outputfile = "output_{}.csv".format(time.strftime("%Y%m%d_%H%M"))

    df_acc[["test_id", "price"]].to_csv(outputfile, index = False)

    print('[{}] All done'.format(time.time() - start_time))

    v = 90
    
    """0.45814, run time ca 15 minutes"""
    """0.45827  28 minutes"""
    

if __name__ == '__main__':
    main()
 
    