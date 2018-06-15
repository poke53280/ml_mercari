
DATA_DIR_PORTABLE = "C:\\avito_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

#Initially forked from Samrat P's kernel here: https://www.kaggle.com/samratp/avito-lightgbm-with-ridge-feature-v-3-0-0-2219/code

#Initially forked from Bojan's kernel here: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2242/code
#improvement using kernel from Nick Brook's kernel here: https://www.kaggle.com/nicapotato/bow-meta-text-and-dense-features-lgbm
#Used oof method from Faron's kernel here: https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867
#Used some text cleaning method from Muhammad Alfiansyah's kernel here: https://www.kaggle.com/muhammadalfiansyah/push-the-lgbm-v19
#Forked From - https://www.kaggle.com/him4318/avito-lightgbm-with-ridge-feature-v-2-0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc



# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

from sklearn.metrics import mean_squared_error
from math import sqrt

def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    
    
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["price"] = np.log(df["price"]+0.001)
    df["price"].fillna(df.price.mean(),inplace=True)

    df["Weekday"] = df['activation_date'].dt.weekday

    df.drop(["activation_date"],axis=1,inplace=True)

    if "image" in df:
        df.drop(["image"],axis=1,inplace=True)

    if "image_top_1" in df:  
        if "image_top_1" in _categorical:
            pass
        else:
            df.drop(["image_top_1"],axis=1,inplace=True)
   

    df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    df['title'] = df['title'].apply(lambda x: cleanName(x))
    df["description"] = df["description"].apply(lambda x: cleanName(x))

    textfeats = ["description", "title"]

    for cols in textfeats:
        df[cols] = df[cols].astype(str) 
        df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
        df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
        df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment)) # Count number of Letters
    
    # Extra Feature Engineering
    df['title_desc_len_ratio'] = df['title_num_letters']/df['description_num_letters']

    return df

"""c"""

def fit_categorical(df, categorical):
    # Fit
    cats = {}

    for col in categorical:
        cats[col] = pd.Categorical(training[col], ordered = False).categories


    return cats

"""c"""

def transform_categorical(df, categorical, catcodes):
    for col in categorical:
        df[col] = pd.Categorical(df[col], categories = catcodes[col], ordered = False)

        df[col] = df[col].cat.codes.astype(np.int)
    return df

"""c"""

def get_col(col_name):
    return lambda x: x[col_name]

"""c"""


const_categorical = ["user_id","region","city","parent_category_name","category_name","user_type", "param_1","param_2","param_3"]

const_russian_stop = set(stopwords.words('russian'))

const_tfidf_para = {
    "stop_words": const_russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}

const_lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 270,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.016,
    'verbose': 0
}  


# TRANSFORM

def transform(df, c_in, vectorizer_in, categorical_in):
    df = transform_categorical(df, categorical_in, c_in)
    X_txt = vectorizer_in.transform(df.to_dict('records'))

    textfeats = ["description", "title"]
    df.drop(textfeats, axis=1,inplace=True)

    return  hstack( [X_txt, df])


_vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=17000,
            **const_tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = const_russian_stop,
            preprocessor=get_col('title')))
    ])


_c = {}

_categorical = []

_is_use_image_top = False


training = pd.read_csv(DATA_DIR + 'train.csv', index_col = "item_id", parse_dates = ["activation_date"])

_categorical = const_categorical

if "image_top_1" in training and _is_use_image_top:
    _categorical.append("image_top_1")

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)

training = preprocess(training)


# FIT

_c = fit_categorical(training, _categorical)
_vectorizer.fit(training.to_dict('records'))

# TRANSFORM

X = transform(training, _c, _vectorizer, _categorical)

tfvocab = _vectorizer.get_feature_names() + training.columns.tolist()

del training
gc.collect()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=23)
        
# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X_train, y_train, feature_name=tfvocab, categorical_feature = _categorical)
lgvalid = lgb.Dataset(X_valid, y_valid, feature_name=tfvocab, categorical_feature = _categorical)

del X, X_train; gc.collect()
    
lgb_clf = lgb.train(const_lgbm_params, lgtrain, num_boost_round=300, valid_sets=[lgtrain, lgvalid], valid_names=['train','valid'], early_stopping_rounds=50, verbose_eval=50)


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))

del X_valid
gc.collect()


### ---------------------------------- CHECKING OUT PSEUDO LABELING -------------------------------------------------------

unlabeled = pd.read_csv(DATA_DIR + 'test_active.csv',  index_col = "item_id", parse_dates = ["activation_date"])
unlabeled = preprocess(unlabeled)
X_unlabeled = transform(unlabeled, _c, _vectorizer, _categorical)

unlabeled_gpred = lgb_clf.predict(X_unlabeled)


unlabeled['idx'] = unlabeled.index

unlabeled.reset_index(inplace=True)

# add column

unlabeled['deal_probability'] = unlabeled_gpred


unlabeled = unlabeled.set_index('idx')

# Drop if exists
unlabeled.drop(["idx"],axis=1,inplace=True)


# SAVE. NOTE COLUMNS ARE PREPROCESSED
unlabeled.to_csv(DATA_DIR + 'unlabeled_w_pred.csv')

# DEL ALL

# LOAD ORIGINALS
test_active = pd.read_csv(DATA_DIR + 'test_active.csv',      index_col = "item_id", parse_dates = ["activation_date"])
training =    pd.read_csv(DATA_DIR + 'train.csv',            index_col = "item_id", parse_dates = ["activation_date"])

# LOAD PREPROCESSED SET WITH PSEUDO LABEL
unlabeled = pd.read_csv(DATA_DIR + 'unlabeled_w_pred.csv',  index_col = "item_id")


# Get test active set with deal_probability.

test_active['deal_probability'] = unlabeled['deal_probability']

# Drop image and image_top_1 on training
training.drop(['image_top_1'], axis=1,inplace=True)
training.drop(['image'], axis=1,inplace=True)


y_train = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)

y_pl    = test_active.deal_probability.copy()
test_active.drop("deal_probability",axis=1, inplace=True)

nTraining = len (training)
nPseudo = int (1 + 0.3 * nTraining)

pl_train = test_active.sample(nPseudo)
pl_train = preprocess(pl_train)

train = preprocess(training)

X_t, X_v, y_t, y_v = train_test_split(train, y_train, test_size=0.10, random_state=23)

# Combine train split and pseudo labeled data. Keep validation chunk apart.



# Fit on train split and pseudo labeled data
X_train = pd.concat([X_t,pl_train])
y_train = pd.concat([y_t, y_pl])


_categorical = const_categorical


_c = fit_categorical(X_train, _categorical)
_vectorizer.fit(X_train.to_dict('records'))

# TRANSFORM

X = transform(train, _c, _vectorizer, _categorical)

X_csr = csr_matrix(X)

X_train = X_csr[:nTraining]
X_pl    = X_csr[nTraining:]

y_train = y[:nTraining]
y_pl    = y[nTraining:]


tfvocab = _vectorizer.get_feature_names() + train.columns.tolist()

del train
gc.collect()



# Add the pl data. Never part of validation.

from scipy.sparse import vstack

X = vstack([X_t, X_pl])
y = pd.concat([y_t, y_pl])

        
# LGBM Dataset Formatting 

lgtrain = lgb.Dataset(X, y, feature_name=tfvocab, categorical_feature = _categorical)

lgvalid = lgb.Dataset(X_v, y_v, feature_name=tfvocab, categorical_feature = _categorical)

del X, X_train; gc.collect()
    
lgb_clf = lgb.train(const_lgbm_params, lgtrain, num_boost_round=3000, valid_sets=[lgtrain, lgvalid], valid_names=['train','valid'], early_stopping_rounds=50, verbose_eval=50)


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_v, lgb_clf.predict(X_v))))

del X_valid
gc.collect()

# [3000]	train's rmse: 0.168191	valid's rmse: 0.219891


testing  = pd.read_csv(DATA_DIR + 'test.csv',  index_col = "item_id", parse_dates = ["activation_date"])

testing.drop(['image_top_1'], axis=1,inplace=True)
testing.drop(['image'], axis=1,inplace=True)


testing = preprocess(testing)
X_test = transform(testing, _c, _vectorizer, _categorical)
lgpred = lgb_clf.predict(X_test)



sample_submission = pd.read_csv(DATA_DIR + 'sample_submission.csv', index_col = 0)
submission = sample_submission.copy()
submission['deal_probability'] = lgpred
submission['deal_probability'] = submission['deal_probability'].clip(0.0, 1.0)

submission.to_csv(DATA_DIR + 'lgsub_mod_pseudo.csv')

# 1. basic lgbm from kernel.
# 'CV': 0.222061209741
# => LB: 2257
#
# 2. pseudo label to 3000
# RMSE: 0.21989061734
# => LB: 2244

#
#
# Levehnstein distance
#
#






