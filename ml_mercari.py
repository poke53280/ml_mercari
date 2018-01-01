
import pandas as pd
import pyximport; pyximport.install()
import gc
import time
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDRegressor
import lightgbm as lgb
import multiprocessing as mp

import random


num_threads = 4

NUM_BRANDS = 1000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000

test.loc[test['category_name'] == 'Women/Sweaters/Hooded'].category_name
train.loc[train['category_name'] == 'Women/Sweaters/Hooded'].category_name

"""Category analysis"""


"""Number of items in each final category"""
q = train.category_name.value_counts()

df = pd.DataFrame(q.index, q)

df.reset_index(inplace=True)

df = df[[0, 'category_name']]

df.columns = ['cat', 'freq']

cat_and_freq = df

cat_and_freq['cat_0'], cat_and_freq['cat_1'], cat_and_freq['cat_2'] = zip(*cat_and_freq['cat'].apply(lambda x: split_cat(x)))

"""Remove full cat"""
cat_and_freq = cat_and_freq [['cat_0', 'cat_1', 'cat_2', 'freq']]

"""Group small categories together/ merge with bigger categories to a minimum of N entries in each remaining category"""
cat_and_freq.loc[cat_and_freq.freq > 2000]
"""=> 151 rows e.g."""

"""Number of largest cat_2 categories vs items contained"""

N = 1000
num_cats = len(cat_and_freq.loc[cat_and_freq.freq > N])
num_items = cat_and_freq.loc[cat_and_freq.freq > N].freq.sum()

num_items_total = cat_and_freq.freq.sum()

print("N = " + str(N) + ", num_cats = " + str(num_cats) + ", num_items = " + str(num_items) + ", " + str(num_items/ num_items_total))


"""

N >   500,  num_cats = 345,  num_items  = 1396885,  0.9462657023942426
N >  1000,  num_cats = 238,  num_items =  1319123,  0.8935888438485633
N >  4000,  num_cats =  88,  num_items  = 1019533,  0.6906431884937624
N >  6000,  num_cats =  62,  num_items  =  890917,  0.6035172550209726
N > 10000,  num_cats =  36,  num_items  =  689400,  0.4670073593965078

"""

cat_and_freq.loc[ (cat_and_freq.cat_0 == 'Handmade')]

cat_and_freq.loc[ (cat_and_freq.cat_0 == 'Handmade') & (cat_and_freq.cat_1 == 'Patterns')]


cat_and_freq.loc[ (cat_and_freq.cat_0 == 'Handmade') & (cat_and_freq.cat_1 == 'Patterns') & (cat_and_freq.P_Cat == 'Drop')].freq.sum()

""" => 48 in Handmade-Patterns to be processed together """



"""Create marker column"""
  

"""
For item in tiny category, take note of category name in _name_ and/or _description."
Or keep it all, just prepare new flat categories
"""




def rmsle(y, y0):
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")


q = 323

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


DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"

DATA_DIR_BASEMENT = "D:\\mercari\\"

DATA_DIR = DATA_DIR_BASEMENT

is_kernel_run = False

is_pumps_only = True


def analyze_run_data():
    f = open(DATA_DIR + "rundata.txt")
    s = f.read()
    f.close()
    l = s.split('\n')
    
    for x in l:
        print(x)



def main():

    start_time = time.time()

    if is_kernel_run:
        train = pd.read_table('../input/train.tsv', engine='c')
        test = pd.read_table('../input/test.tsv', engine='c')
    else:
        train = pd.read_table(DATA_DIR + "train.tsv");
        test = pd.read_csv(DATA_DIR + "test.tsv", sep = "\t", encoding="utf-8", engine = "python");

    print('[{}] Finished to load data'.format(time.time() - start_time))

    
    if is_pumps_only:
        test = test.loc[test.category_name == 'Women/Shoes/Pumps']
        train = train.loc[train.category_name == 'Women/Shoes/Pumps']

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

    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
    watchlist = [d_train, d_valid]
    
    """
    http://lightgbm.readthedocs.io/en/latest/Python-Intro.html
    Specific feature names and categorical features:
    LightGBM can use categorical features as input directly. It doesn’t need to covert to one-hot coding, and is much faster than one-hot coding (about 8x speed-up).

    Note: You should convert your categorical features to int type before you construct Dataset.
    """

    print("Training LGB1")
     

    params = {
        'max_bin' : 255,
        'boosting_type' : 'gbdt',
        'learning_rate': 0.35,
        'application': 'regression',
        'num_leaves': 255,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.5,
        'bagging_freq': 0,
        'nthread': 4,
        'silent' : False,
        'n_estimators' : 400
    }

 


    loops = 231

    while loops > 0:

        start_time = time.time()
     
        


        clf = lgb.LGBMRegressor()

        clf.set_params(**params)

        clf = clf.fit(train_X, train_y)
    
        end_time = time.time()

        y_out = clf.predict(valid_X)

        valid2 = np.expm1(valid_y)
        y_out2 = np.expm1(y_out)

        o = rmsle(y_out2, valid2)

        print("bagging_freq = " + str(params['bagging_freq']) +  " feature_fraction= " + str(params['feature_fraction']) + " bagging_fraction= "
                + str(params['bagging_fraction']) + " RMSLE: " + str(o) + " [{}]s ".format(end_time - start_time))

        loops = loops  -1



    i = 9909

    preds = clf.predict(X_test)

    if is_kernel_run:
        submission['price'] = np.expm1(preds)
        submission.to_csv("submission" + start_time + ".csv", index=False)

if __name__ == '__main__':
    main()
 
v = 90

    