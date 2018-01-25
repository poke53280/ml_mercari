
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

    return isStop

w = 90
   

###############################################################################################
#
#   mem_usage
#
#

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

###############################################################################################
#
#   split_cat
#
#
    
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")


###############################################################################################
#
#   handle_missing_inplace
#
#
    
def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


###############################################################################################
#
#   cutting
#
#

def cutting(dataset):
  

    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


###############################################################################################
#
#   rmsle_func
#
#

def rmsle_func(y, y0):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

q = 90

###############################################################################################
#
#   rmsle_comb
#
#

def rmsle_comb(l):
    counter = 0
    denom = 0
    
    for x in l:
        rmsle = x['rmsle']
        n     = x['n']

        denom = denom + n
        counter = counter + n * rmsle * rmsle

    print("N = " + str(denom))
    
    return np.sqrt (counter/denom)

w = 90


###############################################################################################
#
#   get_XY_Basic
#
#

def get_XY_Basic(df):

  y = np.log1p(df["price"])

  cv0 = CountVectorizer(min_df = 3)
  X_name = cv0.fit_transform(df['name'])
   
  tv = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')

  X_description = tv.fit_transform(df['item_description'])

  lb = LabelBinarizer(sparse_output=True)
    
  X_brand = lb.fit_transform(df['brand_name'])

  X_dummies = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)
  
  X_category = lb.fit_transform(df['category_name'])

  # brand out now:

  X = hstack((X_dummies, X_description, X_name, X_category)).tocsr()

  return {'X': X, 'y':y}


w = 90

###############################################################################################
#
#   get_XY_Advanced
#
#

def get_XY_Advanced(df):

  y = np.log1p(df["price"])

  cv0 = CountVectorizer(min_df = 3)
  X_name = cv0.fit_transform(df['name'])

  lb0 = LabelBinarizer(sparse_output=True)

  print("NUM...")
  s_NUM = noun_ify_spacy(df['item_description'], ['NUM'])

  if is_stop():
      return

  print("SYM...")
  s_SYM = noun_ify_spacy(df['item_description'], ['SYM'])

  if is_stop():
      return

  print("NOUN...")
  s_NN = noun_ify_spacy(df['item_description'], ['NOUN'])

  if is_stop():
      return

  print("PROPN...")
  s_PN = noun_ify_spacy(df['item_description'], ['PROPN'])

  if is_stop():
      return

  tv = CountVectorizer(max_features=1000, min_df = 3)
   
  X_description_NUM = tv.fit_transform(s_NUM)
  X_description_SYM = tv.fit_transform(s_SYM)
  X_description_NN = tv.fit_transform(s_NN)
  X_description_PN = tv.fit_transform(s_PN)

  if is_stop():
      return

  tv = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), stop_words='english')

  X_descriptionFULL = tv.fit_transform(df['item_description'])

  if is_stop():
      return

  lb = LabelBinarizer(sparse_output=True)
    
  X_brand = lb.fit_transform(df['brand_name'])

  X_dummies = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)

  cv = CountVectorizer()
  X_category = cv.fit_transform(df['category_name'])

  X = hstack((X_dummies, X_descriptionFULL, X_description_NUM,
              X_description_SYM, X_description_NN, X_description_PN, X_brand, X_name, X_category)).tocsr()

  return {'X': X, 'y':y}


w = 90

###############################################################################################
#
#   get_by_validation_sequence
#
#

def get_by_validation_sequence(valid_l, epred, id):
    orig_row = valid_l[id]
    u = i[orig_row: orig_row +1]

    desc = u.item_description.values[0]
    name = u.name.values[0]
    brand = u.brand_name.values[0]
    p_in = u.price.values[0]

    p_predicted = epred[id]
   

    s = str(p_in) + ", " + str(p_predicted) + " [" + name + "] [" + brand + "] : " + desc

    return s

w = 90

###############################################################################################
#
#   train1
#
#

def train1(X, y, random, is_output):
   
    idx = list(range(len(y)))

    train_X, valid_X, train_y, valid_y, train_idx, valid_idx = train_test_split(X, y, idx, test_size = 0.1, random_state = random)

    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
        
    watchlist = [d_train, d_valid]
    
    params = { 'learning_rate': 0.01, 'application': 'regression', 'num_leaves': 31, 'verbosity': -1, 'metric': 'RMSE', 'data_random_seed': 1,
                    'bagging_fraction': 0.6, 'bagging_freq': 0, 'nthread': 4, 'max_bin': 255 }

    eval_out = 50

    if is_output:
        eval_out = 35
   
    model = lgb.train(params, train_set=d_train, num_boost_round=9310, valid_sets=watchlist, verbose_eval=eval_out,early_stopping_rounds=400) 
    
    valid_y_pred = model.predict(valid_X)
   
    price_pred = np.expm1(valid_y_pred)
    price_real = np.expm1(valid_y)

    o = rmsle_func(price_pred, price_real)

    y2 = np.power(np.log1p(price_pred)-np.log1p(price_real), 2)

    y2 = y2.values

    if is_output:
        error_dist(y2, 0.1)

    l = (-y2).argsort()

    if is_output:
        for x in l:
            s = get_by_validation_sequence(valid_idx, price_pred, x)
            print (s)


    return o


w = 90

###############################################################################################
#
#   get_cats_contains
#
#

def get_cats_contains(c, txt):
    l = []
    idx = 0
    for x in c:
        if txt in x.lower():
            l.append(idx)

        idx = idx + 1
    return l
w = 90

###############################################################################################
#
#   get_cats_startswith
#
#

def get_cats_startswith(c, txt):
    l = []
    idx = 0
    for x in c:
        if x.lower().startswith(txt.lower()):
            l.append(idx)

        idx = idx + 1
    return l
w = 90


###############################################################################################
#
#   list_cats
#
#

def list_cats(df, cat_IDs, l_first_index):

    acc_amount = 0

    for iCategory in cat_IDs:
        i = get_cat_slice(df, l_first_index, iCategory)

        size = len(i)

        print("Category: " + c[iCategory]+ " size = " + str(size))

        acc_amount = acc_amount + size

    print("#Categories = " + str(len(cat_IDs)) + ", acc_size= " + str(acc_amount))

    
w = 90

cat_IDs = get_cats_startswith(c, 'Wom')

def get_multi_slice(df, cat_IDs, l_first_index):
    
    df_new = pd.DataFrame()

    for iCategory in cat_IDs:
        i = get_cat_slice(df, l_first_index, iCategory)
        df_new = pd.concat([df_new, i])

    return df_new

w = 90

def get_cat_slice(df, l_first_index, iCategory):
    nCategories = len(l_first_index)
    
    assert (iCategory < nCategories)

    start = l_first_index[iCategory]
    
    if iCategory == nCategories -1 :
        passed_end = len(df)
    else:
        passed_end = l_first_index[iCategory +1]

    slice = df[start:passed_end]

    return slice

q = 90

# https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755

def brand_retriever_valking(df, all_brands):

  
   
    # before. 0.51229585402621125

    #51247282934776772

    

    # get to finding!
    premissing = len(df.loc[df['brand_name'] == 'missing'])
    
    def brandfinder(line):
        brand = line[0]
        name = line[1]
        if brand != 'missing':
           return brand
 
        for b in all_brands:
            if (b in name) & (len(b) > 3):
                print("Found brand '" + b + "' in name '" + name + "'")
                return b
        
        return brand

    df['brand_name'] = df[['brand_name','name']].apply(brandfinder, axis = 1)

    found = premissing-len(df.loc[df['brand_name'] == 'missing'])
    
    print(str(found) + " items branded")

     # 0.36354 - pants, with brand retrieval.

     # Better at 6350 iterations with no retrieval. 0.362694 at 9310. no early stop.

     # Tfidf > countvectorizer.same, better after 6350 iterations. 0.362694. same??

     # labelencoder on category. 0.362366

     # again poorer to introduce brand retrieval   0.3639

     # w/o brand alltogheter:  0.3734



def main():
    
    start_time = time.time()
        
    print('[{}] Go'.format(time.time() - start_time))
  
    DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
    DATA_DIR_BASEMENT = "D:\\mercari\\"
    DATA_DIR = DATA_DIR_PORTABLE

    df = pd.read_table(DATA_DIR + "train.tsv");

    df['item_description'].fillna(value='missing', inplace=True)

    df = df.drop(df[(df.price < 3.0)].index)

    # All category brand preprocessing
    df['brand_name'].fillna(value='missing', inplace=True)
    all_brands = set(df['brand_name'].values)
    all_brands.remove('missing')



    df['category_name'].fillna(value='missing', inplace=True)

    NUM_CATEGORIES = 1200
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

    nCategories = len(l_first_index)

    cat_IDs = get_cats_contains(c, 'pants')

    list_cats(df, cat_IDs, l_first_index)


    cat_res = []

    i = get_multi_slice(df, cat_IDs, l_first_index)

    all = df
    df = i

    df = all

    

    brand_retriever_valking(i, all_brands)

    print("Multi-cat, size = " + str(len(i)))

    assert (len(i) > 0)

    d = get_XY_Basic(i)

    y = d['y']
    X = d['X']

    basic_run = 0


    while (basic_run < 1) :

        if is_stop():
            break

        rmsle = train1(X, y, 117 + basic_run, False)
        print("   ===> RMSLE basic = " + str(rmsle))

        d = { 'rmsle' : rmsle, 'n' : len(i) }

        cat_res.append(d)
        basic_run = basic_run + 1

    q = 90

    rmsle_acc = rmsle_comb(cat_res)
    
    q = 90
    

if __name__ == '__main__':
    main()

 
    