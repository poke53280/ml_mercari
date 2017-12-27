
import pandas as pd
import pyximport; pyximport.install()
import gc
import time
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDRegressor
import lightgbm as lgb
import multiprocessing as mp


num_threads = 4

NUM_BRANDS = 1000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000

def rmsle(y, y0):
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


DATA_DIR = "C:\\Users\\T149900\\ml_mercari\\"


def main():

    start_time = time.time()

    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')

    train = pd.read_table(DATA_DIR + "train.tsv");
    test = pd.read_csv(DATA_DIR + "test.tsv", sep = "\t", encoding="utf-8", engine = "python");

    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    
    print('Test shape: ', test.shape)
    
    nrow_train = train.shape[0]

    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()
     
    merge['category_name'].fillna(value='missing', inplace=True)
    merge['brand_name'].fillna(value='missing', inplace=True)
    merge['item_description'].fillna(value='missing', inplace=True)
   
    
    tv0 = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')

    X_name = tv0.fit_transform(merge['name'])

    lb0 = LabelBinarizer(sparse_output=True)

    X_category = lb0.fit_transform(merge['category_name'])
   
    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')

    X_description = tv.fit_transform(merge['item_description'])


    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    
    model = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=1000,
      normalize=False, random_state=101, solver='auto', tol=0.001)
    model.fit(X, y)
    print('[{}] Train ridge completed'.format(time.time() - start_time))
    predsR = model.predict(X=X_test)
    print('[{}] Predict ridge completed'.format(time.time() - start_time))

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.15, random_state = 144) 
    d_train = lgb.Dataset(train_X, label=train_y, max_bin = 8192)
    d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin = 8192)
    watchlist = [d_train, d_valid]
    
    """
    http://lightgbm.readthedocs.io/en/latest/Python-Intro.html
    Specific feature names and categorical features:
    LightGBM can use categorical features as input directly. It doesn’t need to covert to one-hot coding, and is much faster than one-hot coding (about 8x speed-up).

    Note: You should convert your categorical features to int type before you construct Dataset.
    """

    print("Training LGB1")

    start_lgb1_time = time.time()

    params = {
        'learning_rate': 0.78,
        'application': 'regression',
        'num_leaves': 131,
        'verbosity': 0,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.5,
        'bagging_freq': 5,
        'nthread': num_threads,
        'silent' : False
    }

    gridParams = {
        'learning_rate': [0.3, 0.5, 0.7],
        'num_leaves': [20, 31, 140],
        'boosting_type': ['gbdt', 'rf'],
        }


    clf = lgb.LGBMRegressor()

    clf.set_params(**params)

    clf.fit(train_X, train_y)

    y_out = clf.predict(valid_X)

    valid2 = np.expm1(valid_y)
    y_out2 = np.expm1(y_out)

    o = rmsle(y_out2, valid2)

    print("RMSLE: " + str(o))
        
   

    preds = clf.predict(X_test)

    submission['price'] = np.expm1(preds)
    submission.to_csv(DATA_DIR + "submission" + start_time + ".csv", index=False)

if __name__ == '__main__':
    main()
 
v = 90

    