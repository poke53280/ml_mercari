
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

def is_stop():
    f = open(DATA_DIR + "stopfile.txt")
    s = f.read()
    f.close()

    isStop = s[:4] == 'stop'

    return isStop

w = 90
   

    


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
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

q = 90



     d = np.power(np.log1p(y)-np.log1p(y0), 2)

     min = np.argmin(d)
     max = np.argmax(d)


     N = len(y)

     mysum = np.sum(np.power(np.log1p(y)-np.log1p(y0), 2))

     mean =  mysum / N;

     return np.sqrt(mean)


q = 90 



def get_XY_Basic(df):
  y = np.log1p(df["price"])

  cv0 = TfidfVectorizer(ngram_range=(1,14))
  X_name = cv0.fit_transform(df['name'])

  lb0 = LabelBinarizer(sparse_output=True)

  print("NOUN...")
  s_NN = noun_ify_spacy(df['item_description'], ['NOUN'])

  print("PROPN...")
  s_PN = noun_ify_spacy(df['item_description'], ['PROPN'])

  cv = CountVectorizer(stop_words='english')

  X_description_NN = cv.fit_transform(s_NN)
  X_description_PN = cv.fit_transform(s_PN)

   
  tv = TfidfVectorizer(ngram_range=(1, 5))

  X_description = tv.fit_transform(df['item_description'])

  lb = LabelBinarizer(sparse_output=True)
    
  X_brand = lb.fit_transform(df['brand_name'])

  X_dummies = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)

  X = hstack((X_dummies, X_description, X_description_NN, X_description_PN, X_brand, X_name)).tocsr()

  return {'X': X, 'y':y}


w = 90

def get_XY_Advanced(df):

  y = np.log1p(df["price"])

  cv0 = TfidfVectorizer(ngram_range=(1,14))
  X_name = cv0.fit_transform(df['name'])

  lb0 = LabelBinarizer(sparse_output=True)

  print("NUM...")
  s_NUM = noun_ify_spacy(df['item_description'], ['NUM'])

  print("SYM...")
  s_SYM = noun_ify_spacy(df['item_description'], ['SYM'])

  print("NOUN...")
  s_NN = noun_ify_spacy(df['item_description'], ['NOUN'])

  print("PROPN...")
  s_PN = noun_ify_spacy(df['item_description'], ['PROPN'])

  print("VERB...")
  s_VB = noun_ify_spacy(df['item_description'], ['VERB'])

  print("ADJ...")
  s_JJ = noun_ify_spacy(df['item_description'], ['ADJ'])

  tv = TfidfVectorizer(stop_words='english')
   
  X_description_NUM = tv.fit_transform(s_NUM)
  X_description_SYM = tv.fit_transform(s_SYM)
  X_description_NN = tv.fit_transform(s_NN)
  X_description_PN = tv.fit_transform(s_PN)
  X_description_VB  = tv.fit_transform(s_VB )
  X_description_JJ = tv.fit_transform(s_JJ)



  tv = TfidfVectorizer(ngram_range=(1, 17), stop_words='english')

  X_descriptionFULL = tv.fit_transform(df['item_description'])

  lb = LabelBinarizer(sparse_output=True)
    
  X_brand = lb.fit_transform(df['brand_name'])

  X_dummies = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)

  X = hstack((X_dummies, X_descriptionFULL, X_description_NUM,
              X_description_SYM, X_description_NN, X_description_PN, X_description_VB, X_description_JJ, X_brand, X_name)).tocsr()

  return {'X': X, 'y':y}


w = 90



def train1(train_X, valid_X, train_y, valid_y):
   

    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
        
    watchlist = [d_train, d_valid]
    
    params = { 'learning_rate': 0.01, 'application': 'regression', 'num_leaves': 31, 'verbosity': -1, 'metric': 'RMSE', 'data_random_seed': 1,
                    'bagging_fraction': 0.6, 'bagging_freq': 0, 'nthread': 4, 'max_bin': 255 }
   
    model = lgb.train(params, train_set=d_train, num_boost_round=9310, valid_sets=watchlist, verbose_eval=15,early_stopping_rounds=400) 

    
    valid_y_pred = model.predict(valid_X)
   
   
    epred = np.expm1(valid_y_pred)
    evalid = np.expm1(valid_y)

    o = rmsle_func(epred, evalid)

    y2 = np.power(np.log1p(epred)-np.log1p(evalid), 2)

    return o


w = 90




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
    DATA_DIR = DATA_DIR_PORTABLE

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

    while (iCategory < nCategories) & (iProcessed < 200):
       
        i = get_cat_slice(df, l_first_index, iCategory)

        print("Category: " + c[iCategory]+ " size = " + str(len(i)))

        if len(i) > 2000:
            d = get_XY_Advanced(i)
            y = d['y']
            X = d['X']

            d2 = get_XY_Basic(i)
            y2 = d2['y']
            X2 = d2['X']



            basic_run = 0

            l_rmsle = []

            while (basic_run < 3) :
                train_X, valid_X, train_y, valid_y = train_test_split(X2, y2, test_size = 0.1, random_state = 117 + basic_run)
                rmsle = train1(train_X, valid_X, train_y, valid_y)
                print("   ===> RMSLE basic = " + str(rmsle))
                l_rmsle.append(rmsle)
                basic_run = basic_run + 1

            advanced_run = 0

            while (advanced_run < 3) :
                train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.1, random_state = 117 + advanced_run)
                rmsle = train1(train_X, valid_X, train_y, valid_y)
                print("   ===> RMSLE adv = " + str(rmsle))
                l_rmsle.append(rmsle)
                advanced_run = advanced_run + 1

            b = 90

            print(l_rmsle)
            iProcessed = iProcessed +1
           

        iCategory = iCategory + 1

    q = 90
    

if __name__ == '__main__':
    main()

 
    