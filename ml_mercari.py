
import pandas as pd
import pyximport; pyximport.install()
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score

import time


"""Add to below: Find and return a category name with close to desired elements."""

"""Todo: Work in detail on a small category. Post improved kernel."""


"""--------------------------------------------------------------------------------------"""

"""
In Dataframe q, my_cat category, find element closest to having -input- value in brand_name
"""

def get_cutoff(series, count_threshold):
    q = series.value_counts()
    q = q.reset_index()

    cut_index = q.ix[ (q[q.columns[1]] - count_threshold).abs().argsort()[:1]]
    actual_cut_index = cut_index.index[0]
    num_categories = actual_cut_index

    return num_categories

"""--------------------------------------------------------------------------------------"""


"""

In train: brand freq

 500 => 122 items
1000 =>  33 items
1500 =>  14 items
2000 =>   7 items
3000 =>   3 items

"""


NUM_BRANDS = 1500
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

df_train = df_train[df_train.price != 0].reset_index(drop=True)
q = 323


DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"

DATA_DIR_BASEMENT = "D:\\mercari\\"

DATA_DIR = DATA_DIR_BASEMENT


def analyze_run_data():
    f = open(DATA_DIR + "rundata.txt")
    s = f.read()
    f.close()
    l = s.split('\n')
    
    for x in l:
        print(x)


test_full = test      

train_full = train



def train_single_category(train, test):
   
    nrow_train = train.shape[0]

    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    
     
    merge['brand_name'].fillna(value='missing', inplace=True)
    merge['item_description'].fillna(value='missing', inplace=True)
   

    """
    
    http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/
    
    """
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

    
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.11, random_state = 144) 
    
    model = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=20, normalize=False, random_state=101, solver='auto', tol=0.001)

    model.fit(train_X, train_y)

    valid_y_pred = model.predict(valid_X)

    valid_y_pred = np.expm1(valid_y_pred)

    valid_y =  np.expm1(valid_y)

    o = rmsle(valid_y_pred, valid_y)

    print("RMSLE: " + str(o))

    model.fit(X, y)

    y_test = model.predict(X_test)
    y_test = np.expm1(y_test)

    price_series = pd.Series(y_test)
    index_series = pd.Series(test.test_id)

    index_series = index_series.reset_index()

    df = pd.concat([index_series, price_series], axis=1)
 
    df = df.drop(['test_id'], axis = 1)
   
    df.columns = ['test_id', 'price']

    return df

  



i = 9032


def main():
    
    start_time = time.time()

    train = pd.read_table(DATA_DIR + "train.tsv");
    test = pd.read_csv(DATA_DIR + "test.tsv", sep = "\t", encoding="utf-8", engine = "python");

    """Remember fill no category when missing"""

    p = train.category_name.value_counts()
    dfCat = pd.DataFrame(p)

    dfCat = dfCat.reset_index()

    dfCat.columns = ['name', 'counter']

    cat_max = 200
    cat_counter = 0

    df_acc = pd.DataFrame()

    train_acc = 0
    test_acc = 0

    l = ['Women/Suits & Blazers/Blazer']
    t2 = train[~train.category_name.isin(l)]


    while cat_counter < cat_max:
        single_cat = dfCat.name[cat_counter]

        test_single = test.loc[test_full.category_name == single_cat]
        train_single = train.loc[train_full.category_name == single_cat]

        print("'" + single_cat + "'..." + str(len(train_single)) + " item(s)...")

        df = train_single_category(train_single, test_single)
        df_acc = pd.concat([df, df_acc], axis = 0)

        test_acc = test_acc + len(test_single)
        train_acc = train_acc + len (train_single)

        cat_counter = cat_counter + 1

        print('[{}] Finished processing a category'.format(time.time() - start_time))

        print("test_acc = " + str(test_acc) + ", train_acc = " + str(train_acc))


    v = 90


    

if __name__ == '__main__':
    main()
 
v = 90

    