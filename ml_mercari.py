
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import HuberRegressor

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

    if (isStop):
        print("Processing stopped by stop file")

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

def CountVectorize(s):
    cv = CountVectorizer(ngram_range = (1, 17))
    return cv.fit_transform(s)

w = 90

def LabelBinarize(s):
    lb = LabelBinarizer(sparse_output=True)
    return lb.fit_transform(s)

w = 90

def LabelEncode(s):
    lb = LabelEncoder()
    return lb.fit_transform(s)

w = 90

###############################################################################################
#
#   get_XY_Basic
#
#

def get_XY_Basic(df):

  l = []

  l.append( CountVectorize (df['name'])                                         )
  l.append( LabelBinarize  (df['brand_name'])                                   )

  if 'fake_brand' in df:
      print ("fake brand feature enabled")
      l.append (LabelBinarize(df['fake_brand']))

  # DUMMIES -------------------------------------------------------------
  w = 90

  l_dummies = ['item_condition_id', 'shipping']

  if 'qty' in df:
      print("qty feature found")
      l_dummies.append('qty')

  l.append( csr_matrix(pd.get_dummies(df[l_dummies], sparse=True).values))

  l.append( LabelBinarize  (df['category_name']))
  
  X = hstack(l).tocsr()

  y = np.log1p(df["price"])

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

def get_by_validation_sequence(valid_l, epred_lgbm, epred_ridge, epred_huber, id):
    orig_row = valid_l[id]
    u = i[orig_row: orig_row +1]

    desc = u.item_description.values[0]
    name = u.name.values[0]
    brand = u.brand_name.values[0]
    p_in = u.price.values[0]

    p_lgbm = epred_lgbm[id]
    p_ridge = epred_ridge[id]
    p_huber = epred_huber[id]
   

    s = str(p_in) + ", " + str(p_lgbm) +  ", " + str(p_ridge) + ", " + str(p_huber) + " [" + name + "] [" + brand + "] : " + desc

    return s

w = 90


###############################################################################################
#
#   get_by_validation_sequence
#
#
w = 90asdfsdfsdf


def trainSTACK(X, y, splits):

    kf = KFold(n_splits = splits)
    
    nSplits = kf.get_n_splits(X)

    nFold = 0

    l_lgbm = []

    y_stacked = np.zeros(len (y))

    for train_index, valid_index in kf.split(X):
        if is_stop():
            break

        print ("FOLD# " + str(nFold))

        train_X = X[train_index]  
        train_y = y[train_index]

        valid_X = X[valid_index]
        valid_y = y[valid_index]

        price_valid_real = np.expm1(valid_y)

        d_train = lgb.Dataset(train_X, label=train_y)
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        
        watchlist = [d_train, d_valid]
    
        params = { 'learning_rate': 0.001, 'application': 'regression', 'num_leaves': 31, 'verbosity': -1, 'metric': 'RMSE', 'data_random_seed': 1,
                        'bagging_fraction': 0.6, 'bagging_freq': 0, 'nthread': 4, 'max_bin': 255 }

        model_lgbm = lgb.train(params, train_set=d_train, num_boost_round=3110, valid_sets=watchlist, verbose_eval=50, early_stopping_rounds=400)

        preds_lgbm = model_lgbm.predict(valid_X)

        y_stacked[valid_index] = preds_lgbm

        price_lgbm_pred = np.expm1(preds_lgbm)
        o_lgbm = rmsle_func(price_lgbm_pred, price_valid_real)

        print ("LGBM RMSLE: " + str(o_lgbm))
        l_lgbm.append(o_lgbm)

        nFold = nFold + 1

    w = 90
    a_lgbm = np.array(l_lgbm)
    print ("STACK LGBM-RIDGE-HUBER (META: LGBM) RMSLE = " + str (a_lgbm.mean()) + " +/- " + str(a_lgbm.std()))

    return y_stacked

w = 90


###############################################################################################
#
#   trainCV
#
#


X_Backup = X
y_backup = y

X = X_Backup
y = y_backup

def trainCV(X, y, random, splits):
   

    kf = KFold(n_splits = splits)
    
    nSplits = kf.get_n_splits(X)

    nFold = 0

    l_lgbm = []
    l_ridge = []
    l_huber = []

    y_lgbm  = np.zeros(len (y))
    y_ridge = np.zeros(len (y))
    y_huber = np.zeros(len (y))

    for train_index, valid_index in kf.split(X):
        if is_stop():
            break

        print ("FOLD# " + str(nFold))

        train_X = X[train_index]  
        train_y = y[train_index]

        valid_X = X[valid_index]
        valid_y = y[valid_index]

        price_valid_real = np.expm1(valid_y)

        d_train = lgb.Dataset(train_X, label=train_y)
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        
        watchlist = [d_train, d_valid]
    
        params = { 'learning_rate': 0.01, 'application': 'regression', 'num_leaves': 311, 'verbosity': -1, 'metric': 'RMSE', 'data_random_seed': 1,
                        'bagging_fraction': 0.6, 'bagging_freq': 0, 'nthread': 4, 'max_bin': 255 }

        model_lgbm = lgb.train(params, train_set=d_train, num_boost_round=810, valid_sets=watchlist, verbose_eval=50, early_stopping_rounds=400)

        preds_lgbm = model_lgbm.predict(valid_X)

        y_lgbm[valid_index] = preds_lgbm

        price_lgbm_pred = np.expm1(preds_lgbm)
        o_lgbm = rmsle_func(price_lgbm_pred, price_valid_real)

        print ("LGBM RMSLE: " + str(o_lgbm))
        l_lgbm.append(o_lgbm)

        model_ridge = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=50, normalize=False, random_state=101, solver='auto', tol=0.001)

        model_ridge.fit(train_X, train_y)

        preds_ridge = model_ridge.predict(valid_X)
        
        y_ridge[valid_index] = preds_ridge
        
        price_ridge_pred = np.expm1(preds_ridge)
        o_ridge = rmsle_func(price_ridge_pred, price_valid_real)

        print ("RIDGE RMSLE: " + str(o_ridge))
        l_ridge.append(o_ridge)

        model_huber = HuberRegressor(fit_intercept=True, alpha=0.01, max_iter=58, epsilon=363)
        model_huber.fit(train_X, train_y)

        preds_huber = model_huber.predict(valid_X)

        y_huber[valid_index] = preds_huber

        price_huber_pred = np.expm1(preds_huber)
        o_huber = rmsle_func(price_huber_pred, price_valid_real)
    
        print ("HUBER RMSLE: " + str(o_huber))
        l_huber.append(o_huber)

        nFold = nFold + 1
    
    a_lgbm = np.array(l_lgbm)
    a_ridge = np.array(l_ridge)
    a_huber = np.array(l_huber)

    print ("LGBM  RMSLE = " + str (a_lgbm.mean()) + " +/- " + str(a_lgbm.std()))
    print ("RIDGE RMSLE = " + str (a_ridge.mean()) + " +/- " + str(a_ridge.std()))
    print ("HUBER RMSLE = " + str (a_huber.mean()) + " +/- " + str(a_huber.std()))

    return [y_lgbm, y_ridge, y_huber]

w = 90

y = y.values


a, b, c = trainCV(X, y, 119, 5)


#Input a set of predictions a, b, c

#create matrix of all:

X_array = np.column_stack((a, b, c))
X_s = csr_matrix(X_array)

y_s = trainSTACK(X_s, y, 5)


f = LabelEncode(df.category_name)
...




#word baggging with dictionay: test.  train, test and train.


#preds = predsH*w[0] + predsF*w[1] + predsL*w[2] + predsFM*w[3]
    
#To kernel:

    X_array = np.column_stack((predsH, predsF, predsL, predsFM))
    X_s = csr_matrix(X_array)
    
    preds = trainSTACK(X_s, train_y, 11)
    

    submission['price'] = np.expm1(preds)



###############################################################################################
#
#   train1
#
#

def train1(X, y, random, is_output):
    X_Backup = X
    y_backup = y

    idx = list(range(len(y)))

    process_X, holdout_X, process_y, holdout_y, process_idx, holdout_idx = train_test_split(X, y, idx, test_size = 0.1, random_state = random)


    train_X, valid_X, train_y, valid_y, train_idx, valid_idx = train_test_split(process_X, process_y, process_idx, test_size = 0.1, random_state = random)

   

    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
        
    watchlist = [d_train, d_valid]
    
    params = { 'learning_rate': 0.03, 'application': 'regression', 'num_leaves': 31, 'verbosity': -1, 'metric': 'RMSE', 'data_random_seed': 1,
                    'bagging_fraction': 0.6, 'bagging_freq': 0, 'nthread': 4, 'max_bin': 255 }

    eval_out = 50

    if is_output:
        eval_out = 35
   
    model_lgbm = lgb.train(params, train_set=d_train, num_boost_round=6310, valid_sets=watchlist, verbose_eval=eval_out,early_stopping_rounds=400) 
    
    preds_lgbm = model_lgbm.predict(valid_X)

    price_lgbm_pred = np.expm1(preds_lgbm)
    price_valid_real = np.expm1(valid_y)

    o_lgbm = rmsle_func(price_lgbm_pred, price_valid_real)

    print ("LGBM RMSLE: " + str(o_lgbm))


    preds_hold_out_lgbm = model_lgbm.predict(holdout_X)
    price_hold_out_lgbm = np.expm1(preds_hold_out_lgbm)
    price_hold_out_real = np.expm1(holdout_y)

    o_lgbm_holdout = rmsle_func(price_hold_out_lgbm, price_hold_out_real)
    print ("LGBM HOLDOUT RMSLE: " + str(o_lgbm_holdout))

    model_ridge = Ridge(solver = "lsqr", fit_intercept=False)
    model_ridge.fit(train_X, train_y)

    preds_ridge = model_ridge.predict(valid_X)

    price_ridge_pred = np.expm1(preds_ridge)

    o_ridge = rmsle_func(price_ridge_pred, price_valid_real)

    print ("RIDGE RMSLE: " + str(o_ridge))
   

    model_huber = HuberRegressor(fit_intercept=True, alpha=0.01, max_iter=80, epsilon=363)
    model_huber.fit(train_X, train_y)

    preds_huber = model_huber.predict(valid_X)
    price_huber_pred = np.expm1(preds_huber)

    o_huber = rmsle_func(price_huber_pred, price_valid_real)

    print ("HUBER RMSLE: " + str(o_huber))
   


    y2 = np.power(np.log1p(price_lgbm_pred)-np.log1p(price_valid_real), 2)

    y2 = y2.values

    if is_output:
        error_dist(y2, 0.1)

    l = (-y2).argsort()

    # Todo: Display a set of predictions, one for each run model.

    if is_output:
        for x in l:
            s = get_by_validation_sequence(valid_idx, price_lgbm_pred, price_ridge_pred, price_huber_pred, x)
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

###############################################################################################
#
#   get_multi_slice
#
#

def get_multi_slice(df, cat_IDs, l_first_index):
    
    df_new = pd.DataFrame()

    for iCategory in cat_IDs:
        i = get_cat_slice(df, l_first_index, iCategory)
        df_new = pd.concat([df_new, i])

    return df_new

w = 90

###############################################################################################
#
#   get_cat_slice
#
#

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

###############################################################################################
#
#   fake_brand_retriever
#
#

def fake_brand_retriever(df, all_brands):


    def fakebrandfinder(line):
        brand = line[0]
        name = line[1]
        if brand != 'missing':
           return 'missing'
 
        for b in all_brands:
            if (b in name) & (len(b) > 3):
                # print("Found fake brand '" + b + "' in name '" + name + "'")
                return b
        
        return 'missing'

    newColumn = df[['brand_name','name']].apply(fakebrandfinder, axis = 1)

    found = len(newColumn.loc[df['brand_name'] != 'missing'])
    
    print(str(found) + " fake items branded")

    return newColumn




def main():
    
    start_time = time.time()
        
    print('[{}] Go'.format(time.time() - start_time))
  
    DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
    DATA_DIR_BASEMENT = "D:\\mercari\\"
    DATA_DIR = DATA_DIR_PORTABLE

    df = pd.read_table(DATA_DIR + "train.tsv");

    df['item_description'].fillna(value='missing', inplace=True)

    df = df.drop(df[(df.price < 3.0)].index)
    df['brand_name'].fillna(value='missing', inplace=True)

    all_brands = set(df['brand_name'].values)

    all_brands.remove('missing')

    isFakeBrand = False




    q['qty'] = q.item_description.apply(lambda x: SingleStringScanner(x))


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

    l_end_index = []

    for begin in l_first_index:
        if begin == 0:
            pass
        else:
            l_end_index.append(begin)

    l_end_index.append(len(df))

    w = 90

    ranges = list(zip(c, l_first_index, l_end_index))

    categorySet = set (ranges)
    categoryList = list (categorySet)

    # ...

    nCategories = len(l_first_index)

    cat_IDs = get_cats_contains(c, 'elec')

    list_cats(df, cat_IDs, l_first_index)
    
    cat_res = []

    i = get_multi_slice(df, cat_IDs, l_first_index)


    i['qty'] = i.item_description.apply(lambda x: SingleStringScanner(x))





    all = df
    df = i

    df = all

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

 
    