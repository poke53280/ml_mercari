
import pandas as pd
import pyximport; pyximport.install()
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
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


###############################################################################################
#
#   get_XY_Basic
#
#

def get_XY_Basic(df):

  y = np.log1p(df["price"])

  cv0 = CountVectorizer()
  X_name = cv0.fit_transform(df['name'])
   
  tv = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))

  X_description = tv.fit_transform(df['item_description'])

  lb = LabelBinarizer(sparse_output=True)
    
  X_brand = lb.fit_transform(df['brand_name'])

  X_dummies = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)
  
  X_category = lb.fit_transform(df['category_name'])

  X = hstack((X_dummies, X_description, X_brand, X_name, X_category)).tocsr()

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

        model_lgbm = lgb.train(params, train_set=d_train, num_boost_round=3110, valid_sets=watchlist, verbose_eval=0, early_stopping_rounds=400)

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
    
        params = { 'learning_rate': 0.01, 'application': 'regression', 'num_leaves': 31, 'verbosity': -1, 'metric': 'RMSE', 'data_random_seed': 1,
                        'bagging_fraction': 0.6, 'bagging_freq': 0, 'nthread': 4, 'max_bin': 255 }

        model_lgbm = lgb.train(params, train_set=d_train, num_boost_round=810, valid_sets=watchlist, verbose_eval=0, early_stopping_rounds=400)

        preds_lgbm = model_lgbm.predict(valid_X)

        y_lgbm[valid_index] = preds_lgbm

        price_lgbm_pred = np.expm1(preds_lgbm)
        o_lgbm = rmsle_func(price_lgbm_pred, price_valid_real)

        print ("LGBM RMSLE: " + str(o_lgbm))
        l_lgbm.append(o_lgbm)

        model_ridge = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=5000, normalize=False, random_state=101, solver='auto', tol=0.001)

        model_ridge.fit(train_X, train_y)

        preds_ridge = model_ridge.predict(valid_X)
        
        y_ridge[valid_index] = preds_ridge
        
        price_ridge_pred = np.expm1(preds_ridge)
        o_ridge = rmsle_func(price_ridge_pred, price_valid_real)

        print ("RIDGE RMSLE: " + str(o_ridge))
        l_ridge.append(o_ridge)

        model_huber = HuberRegressor(fit_intercept=True, alpha=0.01, max_iter=5800, epsilon=363)
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

y_s = trainSTACK(X_s, y, 11)


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



def fake_brand_retriever(df, all_brands):

   
    # before. 0.51229585402621125

    #51247282934776772

    # get to finding!

    premissing = len(df.loc[df['brand_name'] == 'missing'])
    
    def fakebrandfinder(line):
        brand = line[0]
        name = line[1]
        if brand != 'missing':
           return 'no'
 
        for b in all_brands:
            if (b in name) & (len(b) > 3):
                print("Found fake brand '" + b + "' in name '" + name + "'")
                return b
        
        return 'missing'

    df['fake_brand_name'] = df[['brand_name','name']].apply(fakebrandfinder, axis = 1)

    found = len(df.loc[df['fake_brand_name'] != 'missing'])
    
    print(str(found) + " items branded")

     # 0.36354 - pants, with brand retrieval.

     # Better at 6350 iterations with no retrieval. 0.362694 at 9310. no early stop.

     # Tfidf > countvectorizer.same, better after 6350 iterations. 0.362694. same??

     # labelencoder on category. 0.362366

     # again poorer to introduce brand retrieval   0.3639

     # w/o brand alltogheter:  0.3734


w = 90

l = []

l.append([3.0,   74.4272287063,  'Reserved Show me your mumu tunic dress', 'Free People', ' Show me your mumu tunic dress'])
l.append([206.0, 24.1283899992,  'BNWT Pink Friday Blue Sherpa', 'PINK', 'Brand new with tags pink friday Sherpa. For kit only.'])
l.append([129.0, 14.8767356555,  'Greenidgal ONLY bundle', 'missing', 'This is specifically for Greenidgal. If you see other things that you want me to add to this bundle, please let me know so I can do so when I ship them out. I will be shipping out tomorrow September 7th. Thanks!'])
l.append([172.0, 20.6358229673,  'Bundle', 'Victoria\'s Secret', 'Good condition Gray and lime green Xs 5 tops 4 bottoms'])
l.append([123.0, 15.105401482,   'Fever Cami for Lovely4','Metal Mulisha', 'NWT'])
l.append([186.0, 23.8170222872,  'Bundle', 'Brandy Melville', 'Reserved'])
l.append([167.0, 22.8545650717,  'Bundle for Alicia', 'Under Armour', 'No description yet'])
l.append([7.0,   50.1166123746,  'Taylor only! Bundle!', 'PINK', 'Bundle for Taylor only!'])
l.append([250.0, 39.6490317348,  'Burberry for Sahara *ON HOLD*', 'Burberry', 'Burberry framed heads print top for @SaharaSmith'])
l.append([76.0,  12.03272516,    'MM CLOTHING', 'missing', 'No description yet'])
l.append([84.0,  13.5049856829,  'Two piece see through', 'missing', 'Top xs bottom fit like a small'])
l.append([164.0, 27.368880402,   'Lularoe', 'missing', '3 XS Irma\'s. All brand new. Never been worn. NEW with tags.'])
l.append([206.0, 35.9685678069,  'SUPER RARE STONE COLD FOX SILK HOLY TUBE', 'Stone Fox Swim', 'Silk floral Holy Tube from Stone Cold Fox. Size 1 works best for US 4. Boned inner bodice, silk floral outer, back zip. New with tags.'])
l.append([169.0, 29.3872433192,  '(XS) Victoria\'s Secret Pink', 'PINK', 'In excellent like new condition no flaws. Loose fit.10 items'])
l.append([65.0,  11.0298154399,  'Pizza Slime Gucci Long Sleeve', 'Gucci', 'Size Small. Worn once Pizza Slime Long Sleeve. Popular black long sleeve as seen on Skrillex and Khloe Kardashian. No longer available on their website. Slight cracking to the graphic on the back of the shirt due to defective print the company used. I\'m selling this shirt because they sent me a new one because of the defective ink. Feel free to make an offer!'])
l.append([195.0, 34.8526599481,  'LuLaRoe', 'LuLaRoe', '3 xxs classic tees 2 xxs irmas 3 xs irmas 1 xxs Randy'])
l.append([114.0, 20.8509916941,  'VICTORIA SECRET PINK SHIRT]', 'PINK', 'Rainbow colors pm black long sleeve shirt,in excellent condition, no flaws.PRICE IS FIRM. This is pam don\'t buy.'])
l.append([6.0,   35.744182611,   'ON Hold !!!! Random VS Pink Collection', 'PINK', 'This is what I call a great collection of smalls !!!! So much fun!! You will receive everything in the picture !! Here is a list of everything you will receive --- • 6 destination YourCityYourPink Stickers - Atlanta , Charlotte , Kansas City , Las Vegas , North Jersey , and San Diego • 2 Free With Purchase only in store Sticker one just says " pink " and the other says " running from my problems " • 1 Victoria Secret Pink KeyChain Giftcard balance [rm] • 1 Victoria\'s Secret pink Iron on Dog Patch • 1 pink nation I\'m in pin / button • 1 Associate/ employee pink pen • 1 travel size Cool and Bright Mist • 2 pink travel size Mood mists in pink Stress no More and Pink Zzzz please • 1 package of pink hair ties and bracelets • And 1 Pink Dog logo gift card holder Can include but may or not extra finds while I\'m cleaning :)'])
l.append([3.0,   19.9855776729,  'Blouses are Reserved for buyer', 'Marshall', 'Two top bundle reserved for a buyer already.'])
l.append([81.0,  14.7514133307,  'PINK LOVE PINK L\/S SHIRT bright pink', 'missing', 'RSVD CLEEKHEATHER..Bright pink with Pink wrote in black bold letters with love pink wrote in script below in script writing. Worn twice,didn\'t think i would be selling and I normally cut tags out of my shirts because of the way some just stick up an out or irritate me. Has not been put in dryer was hung to dry. Worn just around house an also wasn\'t planning on selling any of my clothes until my car broke down an needed money to get fixed or this closet wouldn\'t be here. But I have car payments now so I\'m digging through closet an drawers. Price firm. Thanks for looking.plus was to big.HEATHER #1[rm]#2 [rm]#3[rm]#4[rm].#5 [rm] When bundles are made,it\'s pay when done due to a lot of non payments. Thank you. Adding now.'])
l.append([69.0,  12.4600598307,  'Free people led Zeppelin t shirt', 'missing', 'Worn once. Great t shirt!'])
l.append([222.0, 43.1951783544,  'Jane', 'missing', 'Bundle 226 5/19'])
l.append([81.0,  15.5007073445,  'Palace P45 T-Shirt Grey Marl (Med)', 'Supreme', 'Brand New Palace Skateboards P45 T-Shirt in hand. Bought off the website'])
l.append([110.0, 22.3030934272,  'Bundle for Allanna', 'missing', 'Bundle'])
l.append([93.0,  18.819091727,   'Dkny NWT blouse', 'DKNY', 'Prana yoga long sleeve, white tank, and black beaded tank.'])
l.append([45.0,   9.0699913812,  'Have a nice death.', 'missing', 'Perfect condition.'])


# Bath bombs

b = []

b.append([103.0, 17.6706128964, 'Glitterdollz', '8 items Sex bomb, Golden Wonder bomb, Jester, Butter Bear, Stardust, experimenter, comforter and mistletoe.'])
b.append([124.0, 26.9171780002, 'Lush bundle *RESERVED*', 'All brand new. Includes everything listed in both pictures.'])
b.append([125.0, 28.9232761716, 'Lush Bundle on hold','FOUR brand new reusable bars. A santa in space gift box, a large bubbly shower gel and a night before Christmas gift box'])
b.append([57.0, 15.6284718555, 'Heated Foot Spa', 'Used just a few times - awesome!! Just don\'t have room for it anymore. Purchased brand new from Amazon. Two massage rollers, heating therapy, oxygen bubbles massage, water fall & wave massage. Digital temperature and time control. LED display.'])
b.append([40.0, 10.7961140316, '5 Error 404 Lush Bath Bombs', 'These are the new scent that was sold during black Friday. Wrapped in plastic wrap to keep them fresh.'])
b.append([35.0, 9.52420520714, 'Bath and body works', 'New and never been used'])
b.append([3.0, 12.5056078215, 'For Cassandra', 'Includes One 5oz Bath Bomb Each batch is made from the best organic ingredients and tested for quality •just place in the bath and relax •smells great and makes your skin super soft •Perfect gifts for Valentines Day coming up ✨Baking Soda, Citric Acid, Corn Starch, coconut oil, espom salts, food coloring, glitter, and essential oils (exotic jasmine) ✨tags: bath bomb, gift for her, black magic, Galaxy dust, glitter, LUSH, bath and body works, relax, zen, bath, home accessories, Valentine\'s Day gifts ✨want to bundle more? Just comment and I\'ll make you a listing. 1 for [rm] 2 for [rm] 3 for [rm] 4 for [rm] 5 for [rm] All with free next day shipping and tracking'])
b.append([115.0, 34.2754220551, '4 bath & body works sets', ' Brand new bath & body works set includes 8oz fragrance mist, 8oz body cream, 8.4oz shower gel & 8.4oz body lotion. Pet/smoke free. **price is firm/no trades/no free ship/no holds/will not separate** Cocktail dress Bourbon strawberry & vanilla Poolside coconut colada Golden pear & brown sugar'])
b.append([7.0, 24.0414371843, 'Lush sample empty containers bags bundle', 'Lush sample empty containers & bags bundle. BUNDLES ARE THE BEST☺☺ SAVE $$$ check out my other items in my closet Lush 15 gift shopping bags and gift 10 sample empty containers bundle bags are different size and great condition. The sample containers are good to give as gifts when buyers buy your lush products. presentation in selling you lush products will give you a 5 star rating. these would be great if you are selling Lush items such as perfumes bath bombs and other lush cosmetics. perfect for gift wrapping birthday and holiday present. they are in great condition Lush set Lush cosmetics Lush gifts Lush bath bomb'])

b.append([9.0, 29.4772961577, 'On hold for megalooper', 'Love spell Snowcastle Cherry blossom bubble bar Orange blossom Lux pud Havana sunrise sample'])
b.append([7.0, 22.7906094201, 'On hold for megalooper', 'Brand new easter collection scrub! Shares its scent with honey I washed the kids but is a bit more coconut scented to my nose, similar to Buffy or king of skin :)'])
b.append([35.0, 11.4084350349, 'Full size exotic coconut', 'Brand new never used, full sized products. Shower gel and body splash. Smoke free home. [rm] each or [rm] sold together. I ship on Tuesdays as its my only day off. Pricing is firm! On this item! Free shipping on this item. Selling together, only'])
b.append([65.0, 22.0955372783, 'Lush Kitchen 7 pc. Citrus Spring/Rhea', 'All made fresh in the Kitchen this month Sunny Citrus soap Somewhere Over The Rainbow Soap Over & Over Bomb Ups-A-Daisy bath bomb The Sicilian Mum bomb Your Mother Should Know [rm] plus [rm] shipping'])
b.append([17.0, 5.3549440164, 'PINK bath Bomb', 'Calming Vanilla'])
b.append([9.0, 26.6029652528, 'Body wash bundle', 'This includes (3) 1.2 oz herbal essences body wash (1) 1.7oz old spice Fiji body wash (1) Nivea touch of happiness body wash **NO FREE SHIP**'])
b.append([84.0, 29.9873399684, 'Reserved Lush Bundle', 'RESERVED Free USPS Shipping Brand new and never used 1 x The rough with the smooth 2 x unicorn horn 2 x candy mountain 2 x French kiss 1 x butter bear 1 x Santa\'s belly'])

# Shoes - no or few bundles for inverse check
s = []

s.append([100.0, 20.2953771963, 'Luchesse horn back crocs.', 'missing', 'Women\'s 10. Men\'s 8.5'])
s.append([120.0, 25.1165618361, 'Chanel !!!NEW!!!', 'Chanel', 'NEW CHANEL NO SCRATCHES'])
s.append([55.0, 16.0658097137, 'Vince Verell Slip-On Sneakers 6', 'Vince', 'Great Condition. Only worn 1/2 times. Back of shoe has a little rip from where my sister\'s foot got caught. Not noticeable when wearing. Pictured above. Smoke/Pet Free.'])
s.append([125.0, 40.9778933657, 'Burberry Espadrilles', 'Burberry', 'No description yet'])
s.append([5.0, 16.6341695176, 'Sperry boat shoes Ladies size 8', 'Sperry', 'Ladies animal print Sperry boat shoes Ladies size 8. Gently worn.'])
s.append([9.0, 28.3821428738, 'Pink suede UGGS', 'UGG Australia', 'Great condition, ugg slip ons with pink suede lined with fur inside'])
s.append([9.0, 27.2920805847, 'Pink bow slide on', 'missing', 'This pink bow slip on are brand new never worn. They are a size 7-8 but, can easily fit a 8.5. Very girly and cute'])
s.append([9.0, 24.1096211128, 'Pedro reserved', 'missing', 'In women\'s size 6'])
s.append([81.0, 31.9562287335, 'Black Uggs', 'UGG Australia', 'Black uggs slippers that you can wear anywhere'])

s.append([46.0, 17.9845374972, 'Ariat women\'s Cruiser slip on shoes', 'Ariat', 'Good condition, minimal wear. Very comfortable. Retail for [rm] new.'])
s.append([9.0, 22.1907965682, 'Size 8 Danskin slip on sneakers', 'Danskin', 'Barely worn Size 8 Dansko brand slip ons with memory foam No box'])
s.append([5.0, 12.8517391783, 'moccasins.', 'missing', 'No description yet'])


n = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "elleven", "twelve"]


#text cut after:

# . Please 
# .Free shipping
# .No free shipping

# Check out...
# Will ship....
# Shipping
# Free priority mail
# bundle to save

import re

def StringScanner(s):

    #brands I have include....

    # Preprocess:
    # remove n in one
    # two in one
    # Sz.7.5
    # 1960 - 2020 - likely year drop here or in post?

    #number am
    #number pm
    #1st, 2nd

    #matt27  - drop
    #stone   - drop (one)

    #...both ...or ...each... X and Y.
    # ...both for or each for... rm or number


    # will combine

    # Ships fast!

    #  10 yrs, year years

    # in total ....100 cards...

    # one side, two-sided 

    # all/included/on the in the picture

    # set of... noun + s.

    # 4/25 green card
    # Card #84 -124

    # Holo Machamp 9/101

    # look for item sum: [('10', 'pajama'), ('3', 'pajama'), ('7', 'pajama')]. Almost certain bundle.

    # Freely message

    # please do not

    # prices are....

    # boy's medium 10/12 brand new 

    # Sizes medium 10/12

    # and H&M, Crazy 8.

    # size small 5/6

    # Sizes 12-16 

    # Thank you for 

    # P.S.

    # do not buy

    # please let me

    # no description yet => na

    # Price includes shipping 

    #model A1218. Worges great,  - '1219. works'

    # 7", 10.1"
    # 1 case is [rm] 2 cases is [rm] 3 cases is [rm] 

    # gen, generation 3rd gen. gen 3. Generation 3. first gen

    #GB, gb, gigs, gig.

 
    # 48MB/s => '48', 'mb'

    # USB 3.0


    s = re.sub("men's \d+\.?\d*", "men's", s)
    s = re.sub("mens \d+\.?\d*", "men's", s)
    s = re.sub("womens \d+\.?\d*", "women's", s)

    s = re.sub("size \d+-\d+", "size", s)
    s = re.sub("size \d+\.?\d*", "size", s)

    # Remove size number  'Size 8' , size 8.5 size 7-8
    # Remove women's 10. Men's 8.5
    # Remove digit times.  worn two times. worn one time. Keep time/times.



    print(s)

    s = s.lower()
    letter_digits = "|one|two|three|four|five|six|seven|eight|nine|ten"

    regex_number_and_item = "(\d+\.?\d*" + letter_digits + ")\s*([a-zA-Z]+)"

    regex_number_in_parantheses = "\((\d+)\)"

    regex_word_signal = "bundle|all|everything|and|collection"

    regex_word_antisignal = "will bundle|each"

    m1 = re.findall(regex_number_and_item, s)
    m2 = re.findall(regex_number_in_parantheses, s)

    print (m1)
    print (m2)

    # remove qty + time, qty + times
    


    print ("-----------")

w = 90

def SeriesScanner(s):
    for x in s:
        StringScanner(x)

w = 90

def QtyScanner(l):
    for x in l:
        StringScanner(x[4])
        



w = 90


QtyScanner(l)

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

    cat_IDs = get_cats_contains(c, '/')

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

 
    