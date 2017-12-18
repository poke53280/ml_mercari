
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
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb
import multiprocessing as mp

from collections import Counter
import matplotlib.pyplot as plt


num_threads = mp.cpu_count()

NUM_BRANDS = 1000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000

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


DATA_DIR = "C:\\Users\\T149900\\mercari\\"

TEXT_DIR = "C:\\Users\\T149900\\Documents\\Visual Studio 2017\\Projects\\ml_mercari\\"

text_files = [TEXT_DIR + "xeno.txt", 
              TEXT_DIR + "gt.txt"]


"""concat all else against one name"""


j = 90

def last_word_in_name(df, cat):
    list = []

    if cat == "":
        nameSeries = df.name
    else:
        nameSeries = df.loc[df.category_name == cat].name
    
    a = nameSeries.values

    idx = 0
    while idx < len(a):
        str = a[idx]
        w = str.split()
        last_word = w.pop().lower()[:5]
        list.append(last_word)
        idx += 1

    return list


p = 90

b3 = [val for val in b1 if val in b2]

def to_freq(total_count, values):
    freq = []
    for x in values: freq.append(x/total_count)
    return freq



def get_common_words(l1, l2):

    counts1 = Counter(l1)
    counts2 = Counter(l2)

    labels1, values1 = zip(*counts1.items())
    labels2, values2 = zip(*counts2.items())


    values1 = to_freq(len(l1), values1)
    values2 = to_freq(len(l2), values2)


    indSort = np.argsort(values1)[::-1]
    labels1 = np.array(labels1)[indSort]
    values1 = np.array(values1)[indSort]
    indexes1 = np.arange(len(labels1))

    indSort = np.argsort(values2)[::-1]
    labels2 = np.array(labels2)[indSort]
    values2 = np.array(values2)[indSort]
    indexes2 = np.arange(len(labels2))

q = 324


def text_init():
    train = pd.read_table(DATA_DIR + "train.tsv");
    test = pd.read_csv(DATA_DIR + "test.tsv", sep = "\t", encoding="utf-8", engine = "python");

    full = train.append(test)

    cat0 = "Women/Athletic Apparel/Pants, Tights, Leggings"
    cat1 = "Handmade/Glass/Bottles"
    cat2 = "Women/Tops & Blouses/T-Shirts"
    cat3 = "Women/Tops & Blouses/Tank, Cami"
    cat4 = "Women/Tops & Blouses/Blouse"
    cat5 = "Electronics/Video Games & Consoles/Games"
    cat6 = "Electronics/Cell Phones & Accessories/Cases, Covers & Skins"

    cat7 = "Beauty/Makeup/Face"
    cat8 = "Beauty/Makeup/Lips"
    cat9 = "Beauty/Makeup/Makeup Palettes"


    l1 = last_word_in_name(full, cat1)
    l2 = last_word_in_name(full, cat2)
    l3 = last_word_in_name(full, cat3)
   

    long_string = nameSeries.to_string(index = False)

    corpus_data = [long_string]

    cv = CountVectorizer()

    cv.fit(corpus_data)

    l = cv.get_feature_names()

    vector_corpus = cv.transform(corpus_data)

def text_test(text_str1):
    vector2 = cv.transform([text_str1])

    nz_vec2 = np.nonzero(vector2)

    nz_vec2[1]

    hit_array = nz_vec2[1]


    for idx in np.nditer(hit_array):
        word = l[idx]
        c_c = vector_corpus[0, idx]
        c_t = vector2[0, idx]
        print (word + " " + str(c_c) + " " + str(c_t))


a = 9323
train.loc[train.category_name == "Handmade/Glass/Bottles"]



def plot_test(word_list):
    
    counts = Counter(word_list)
    labels, values = zip(*counts.items())
    indSort = np.argsort(values)[::-1]
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    indexes = np.arange(len(labels))
    bar_width = 0.35

    max_num = 50

    indexes = indexes[:max_num]
    values = values[:max_num]
    labels = labels[:max_num]

    plt.bar(indexes, values)

    plt.xticks(indexes + bar_width, labels)
    plt.show()



q = 90

def text_standout():
     tv = TfidfVectorizer(input = 'content', max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')

     cv = CountVectorizer()
     
     long_string = train.name.to_string() + test.name.to_string()

     corpus_data = [long_string]

     cv.fit(corpus_data)

     l = cv.get_feature_names()

     vector_corpus = cv.transform(corpus_data)


    
     text_str1 = "nwt never worn michael kros extra blouse"

     vector2 = cv.transform([text_str1])

     nz_vec2 = np.nonzero(vector2)

     nz_vec2[1]

     hit_array = nz_vec2[1]


     for idx in np.nditer(hit_array):
        word = l[idx]
        c_c = vector_corpus[0, idx]
        c_t = vector2[0, idx]
        print (word + " " + str(c_c) + " " + str(c_t))
        


        
a = 93



def main():

    start_time = time.time()

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
    
    print("Training LGB1")

    start_lgb1_time = time.time()

    params = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': num_threads,
        'max_bin' : 8192
    }

    model = lgb.train(params, train_set=d_train, num_boost_round=8000, valid_sets=watchlist, \
    early_stopping_rounds=250, verbose_eval=1000) 
    predsL = model.predict(X_test)

    print('[{}] lgb 1 timing'.format(time.time() - start_lgb1_time))
    
    print('[{}] Predict lgb 1 completed.'.format(time.time() - start_time))

    print("Training LGB2")

    params2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 140,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': num_threads,
        'max_bin' : 8192
    }
    
    train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101) 
    d_train2 = lgb.Dataset(train_X2, label=train_y2)
    d_valid2 = lgb.Dataset(valid_X2, label=valid_y2)
    watchlist2 = [d_train2, d_valid2]

    start_lgb2_time = time.time()

    model = lgb.train(params2, train_set=d_train2, num_boost_round=8000, valid_sets=watchlist2, \
    early_stopping_rounds=150, verbose_eval=1000) 
    predsL2 = model.predict(X_test)

    print('[{}] lgb 2 timing.'.format(time.time() - start_lgb2_time))

    print('[{}] Predict lgb 2 completed.'.format(time.time() - start_time))

    preds = predsR*0.3 + predsL*0.35 + predsL2*0.35

    submission['price'] = np.expm1(preds)
    submission.to_csv(DATA_DIR + "submission" + start_time + ".csv", index=False)

if __name__ == '__main__':
    main()

def find_name(counter, text):
    sumx = sum(counter.values())

    words = text.split()

    words = [x.lower()[:5] for x in words]



    freq = []

    for x in words:
        f = counter[x]/ sumx
        freq.append(f)

    return freq     
v = 90

    