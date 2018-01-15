
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
import lightgbm as lgb

"""  
Text processing read:
    http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/

Categorical:
    http://pbpython.com/categorical-encoding.html
"""

"""Add to below: Find and return a category name with close to desired elements."""

"""Todo: Work in detail on a small category. Post improved kernel."""


"""--------------------------------------------------------------------------------------"""


def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    elif isinstance(pandas_obj,pd.Series):
        usage_b = pandas_obj.memory_usage(deep=True)
    else:
        assert(0)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

q = 90

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

   

    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'



def rmsle_func(y, y0):


     d = np.power(np.log1p(y)-np.log1p(y0), 2)

     min = np.argmin(d)
     max = np.argmax(d)


     N = len(y)

     mysum = np.sum(np.power(np.log1p(y)-np.log1p(y0), 2))

     mean =  mysum / N;

     return np.sqrt(mean)


 




w = 99



def get_XY(df):
  y = np.log1p(df["price"])

  cv0 = TfidfVectorizer(ngram_range=(1,14))
  X_name = cv0.fit_transform(df['name'])

  lb0 = LabelBinarizer(sparse_output=True)
   
  tv = TfidfVectorizer(ngram_range=(2, 17))

  X_description = tv.fit_transform(df['item_description'])

  lb = LabelBinarizer(sparse_output=True)
    
  X_brand = lb.fit_transform(df['brand_name'])

  X_dummies = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)

  X = hstack((X_dummies, X_description, X_brand, X_name)).tocsr()

  return {'X': X, 'y':y}


w = 90



def train2(train_X, valid_X, train_y, valid_y):
  
    clf = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=410, max_bin=255, bagging_fraction= 0.8, nthreads = 4, min_child_samples=10)

    clf.fit(train_X, train_y, early_stopping_rounds = 30, eval_set = [(valid_X, valid_y)], eval_metric = 'rmse')

    #Check prediciton on validation set

    y = clf.predict(valid_X)

    y = np.expm1(y)
    valid_y_pred = np.expm1(valid_y)

    o = rmsle_func(y, valid_y_pred)

    print(o)


w = 90


def train(train_X, valid_X, train_y, valid_y):
   

    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
        
    watchlist = [d_train, d_valid]
    
    params = { 'learning_rate': 0.1, 'application': 'regression', 'num_leaves': 31, 'verbosity': -1, 'metric': 'RMSE', 'data_random_seed': 1, 'bagging_fraction': 0.5, 'nthread': 4, 'max_bin': 255 }
   
    model = lgb.train(params, train_set=d_train, num_boost_round=110, valid_sets=watchlist, verbose_eval=25) 


w = 90


def train_single_category(X, y, random):
   
    start_time = time.time()

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.11, random_state = random)

    train(train_X, valid_X, train_y, valid_y)

    valid_y_pred = model.predict(valid_X)
    valid_y_pred = np.expm1(valid_y_pred)
    valid_y =  np.expm1(valid_y)
    o = rmsle_func(valid_y_pred, valid_y)

    print('[{0:4.0f}] Model run complete'.format(time.time() - start_time))
   
    print(o)

    return o



"""sort all categories. Find start and stop for all categories. work on slices into arrays"""




i = 323
  


#secure multiparty communication. a/b.
#brukerreise- NN



q = 90

def get_cat_slice(df, c, iCategory):
    nCategories = len(c)
    
    assert (iCategory < nCategories)

    start = c[iCategory]
    
    if iCategory == nCategories -1 :
        passed_end = len(df)
    else:
        passed_end = c[iCategory +1]

    slice = df[start:passed_end]

    return slice

q = 90



def main():
    
    start_time = time.time()
        
    print('[{}] Go'.format(time.time() - start_time))
  
    DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
    DATA_DIR_BASEMENT = "D:\\mercari\\"
    DATA_DIR = DATA_DIR_BASEMENT

    df = pd.read_table(DATA_DIR + "train.tsv");

    df['brand_name'].fillna(value='missing', inplace=True)

    df['item_description'].fillna(value='missing', inplace=True)


    df = df.drop(df[(df.price < 1.0)].index)

    df['category_name'].fillna(value='missing', inplace=True)

    NUM_BRANDS = 4500
    NUM_CATEGORIES = 1200

    pop_brand = df['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    df.loc[~df['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'

    pop_category0 = df['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]


    df.loc[~df['category_name'].isin(pop_category0), 'category_name'] = 'missing'


    df.category_name = df.category_name.astype('category')

    """sort by category. retrieve start index for each category"""

    df = df.sort_values(by = 'category_name')

    df.reset_index(inplace = True)

    df.category_name = df.category_name.cat.as_ordered()

    """ordered category list"""
    c = df.category_name.cat.categories

    l_first_index = []

    for c_value in c:
        x = df.category_name.searchsorted(c_value)
        l_first_index.append(x[0])

    df = df.drop(['category_name'], axis = 1)

    iCategory = 0

    iProcessed = 0

    nCategories = len(l_first_index)

    while (iCategory < nCategories) & (iProcessed < 1):
        print("Category: " + c[iCategory])
        i = get_cat_slice(df, l_first_index, iCategory)

        if len(i) > 2000:
            d = get_XY(i)
            y = d['y']
            X = d['X']

            rmsleA = train_single_category(X,y, 144)
            print("   ===> RMSLEA = " + str(rmsleA))

            rmsleB = train_single_category(X,y, 202)
            print("   ===> RMSLEB = " + str(rmsleB))

            rmsleC = train_single_category(X,y, 90)
            print("   ===> RMSLEC = " + str(rmsleC))

            iProcessed = iProcessed +1
           

        iCategory = iCategory + 1


if __name__ == '__main__':
    main()

 
    