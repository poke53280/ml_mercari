
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

Categorical:
    http://pbpython.com/categorical-encoding.html
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

    NUM_BRANDS = 4500
    NUM_CATEGORIES = 1200

    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'

    pop_category0 = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['category_name'].isin(pop_category0), 'category_name'] = 'missing'

    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'

def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


def rmsle_func(y, y0):
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


 




w = 99



def train_single_category(train, test):
   
    start_time = time.time()

    print('[{0:4.0f}] Start single'.format(time.time() - start_time))

    nrow_train = train.shape[0]

    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])

   
    
    cv0 = TfidfVectorizer(ngram_range=(1,5))

    X_name = cv0.fit_transform(merge['name'])

    lb0 = LabelBinarizer(sparse_output=True)
   
    tv = TfidfVectorizer(ngram_range=(1, 5))

    X_description = tv.fit_transform(merge['item_description'])

    lb = LabelBinarizer(sparse_output=True)
    
    X_brand = lb.fit_transform(merge['brand_name'])

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_name)).tocsr()

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.31, random_state = 144)

    print('[{0:4.0f}] Start GBM'.format(time.time() - start_time))

    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
        
    watchlist = [d_train, d_valid]
    
    params = {
        'learning_rate': 0.1,
        'application': 'regression',
        'num_leaves': 255,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4,
        'max_bin': 8192
    }
   
    model = lgb.train(params, train_set=d_train, num_boost_round=5000, valid_sets=watchlist, early_stopping_rounds=250, verbose_eval=500) 
    valid_y_pred = model.predict(valid_X)
    valid_y_pred = np.expm1(valid_y_pred)
    valid_y =  np.expm1(valid_y)
    o = rmsle_func(valid_y_pred, valid_y)

    print('[{0:4.0f}] Model run complete'.format(time.time() - start_time))
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




x = pd.Categorical(['apple', 'bread', 'beer', 'cheese', 'milk' ])
x = s.searchsorted(my_cat[2])




i = 323
  

def preprocess(merge, start_time):
    
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
 
    
    
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))





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

    full_train = full_train.drop(full_train[(full_train.price < 1.0)].index)
    nrow_train = full_train.shape[0] 

    y = np.log1p(full_train["price"])

    full_train.drop(["price"], inplace = True, axis = 1)

    len_train = len(full_train)

    full_train = full_train.rename(columns={'train_id': 'id'})
    full_test =  full_test.rename(columns={'test_id': 'id'})

    merge: pd.DataFrame = pd.concat([full_train, full_test])

    del full_train
    del full_test
    gc.collect()


    preprocess(merge, start_time)
   

    p = merge.category_name.value_counts()

    dfCat = pd.DataFrame(p)

    if isReverse:
        dfCat = dfCat.reindex(index=dfCat.index[::-1])

    dfCat = dfCat.reset_index()

    dfCat.columns = ['name', 'counter']

    cat_max = 3

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

    train2 = full_train[~full_train.category_name.isin(processed_cats)]
    test2 =  full_test[~full_test.category_name.isin(processed_cats)]

    df = train_standard(train2, test2)
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
 
    