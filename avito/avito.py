

import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

from nltk.corpus import stopwords 
import lightgbm as lgb

# from wordbatch.models import FM_FTRL

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
    'verbose': 1
}  


#from wordbatch.models import FM_FTRL

# Prev avito winners:
# stemming, lemmatization, transliteration
# Distances different similarity features between title-title, title-description, title-json like:
# Cosine distance,
# Levenshtein (see: study_Lavrikov_Ridge.py)
# Jaccard,
# NCD,
# etc
#
# http://blog.kaggle.com/2016/08/24/avito-duplicate-ads-detection-winners-interview-1st-place-team-devil-team-stanislav-dmitrii/
#
#
#
# Mercari - study_konstantin_pavel.py, 
# 
# 
# 
# study_mercaring_2nd_place.py
#
# FM_FTRL - Mercari
#
# LGBM from active kernels.
#
# Basic image analysis from forums.
# Toxic?
#
#
# https://github.com/alexeygrigorev/avito-duplicates-kaggle
#
#
#
# From pavel/konstantin : https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
#
#

# Todo: What to train for long. Logging along the way. Stop on key or file. Partial fit. Validation, 
# cross validation along the way, alternate between models.
# 'break into' and reconfigure and train on with new parameters.
#

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')


#def fit_predict_FM(xs, y_train) -> np.ndarray:
#    X_train, X_test = xs
#
#    model = FM_FTRL(alpha=0.035, beta=0.001, L1=0.00001, L2=0.15, D=X_train.shape[1],
#                alpha_fm=0.05, L2_fm=0.0, init_fm=0.01,
#                D_fm=100, e_noise=0, iters=3, inv_link="identity", threads=4)
#
#    import scipy
#    X_mine = scipy.sparse.csr_matrix( (len (y_train),4), dtype=np.float64 )
#
#
#    model.fit(X_train.astype(np.float64), y_train.astype(np.float64))
#    model.fit(X_mine, y_train.astype(np.float64))
#
#    return model.predict(X_test)[:, 0]

def fit_predict_LGBM(xs, y_train) -> np.ndarray:
    X_train, X_test = xs

    y_1D = y_train.ravel()

    lgtrain = lgb.Dataset(X_train, y_1D)
    model = lgb.train(const_lgbm_params, lgtrain, num_boost_round=3, verbose_eval=1)

    return model.predict(X_test)[:, 0]


def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)  # Sigmoid gives poorer CV.
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(7):
            print(f"Epoch {i + 1}...")
            model.fit(x=X_train, y=y_train, batch_size=2**(7 + i), epochs=1, verbose=0)
        return model.predict(X_test)[:, 0]


def create_name_pipeline():
    l = []
    l.append( ('td_name', CountVectorizer(max_features=170000, stop_words = const_russian_stop, ngram_range=(1, 2))))
    return Pipeline(l)


def create_text_pipeline():
    l = []
    l.append ( ('td_text', Tfidf(max_features=170000, token_pattern='\w+', ngram_range=(1, 3))))

    return Pipeline(l)
"""c"""

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    
    df.loc[:, "price"] = np.log(df["price"]+0.001)

    df["price"].fillna(df.price.mean(),inplace=True)

    df.loc[:, 'image_top_1'] = df['image_top_1'].fillna(0).astype(str)
    
    df['name'] = df['title'].fillna('') # + ' ' + df['brand_name'].fillna('')

    df.loc[:, 'text'] = df['description'].fillna('') + " " + df['name'].fillna('')

    feature = ['image_top_1', 'region', 'city',  'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type']
    prefix  = ['image_'     , 'reg_',   'city_',  'p_cat_'             , 'cat_',          'p1_',     'p2_'    , 'p3_',     'ut_']

    for (f, p) in zip (feature, prefix):
        df.loc[:, 'text'] = df['text'] + ' ' + p + df[f].fillna('')
    """c"""
    
    return df[['name', 'text', 'price']]

DATA_DIR_PORTABLE = "C:\\avito_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

const_russian_stop = set(stopwords.words('russian'))

train = pd.read_csv(DATA_DIR + 'train.csv', index_col = "item_id", parse_dates = ["activation_date"])

y_scaler = StandardScaler()

cv = KFold(n_splits=20, shuffle=True, random_state=42)
train_ids, valid_ids = next(cv.split(train))
train, valid = train.iloc[train_ids], train.iloc[valid_ids]

y_train = y_scaler.fit_transform(train['deal_probability'].values.reshape(-1, 1))


vectorizer = make_union(on_field('name', create_name_pipeline()),
                        on_field('text', create_text_pipeline()),
                        on_field(['price'], FunctionTransformer(to_records, validate=False), DictVectorizer()))

X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)

print(f'X_train: {X_train.shape} of {X_train.dtype}')

del train
    
X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)

print(f'X_valid: {X_valid.shape} of {X_valid.dtype}')


############### PERSISTENCE POINT ###################################
#
# X_train
# X_valid
#
#

from scipy import sparse

sparse.save_npz(DATA_DIR + "X_train_avitopy.npz", X_train)
sparse.save_npz(DATA_DIR + "X_valid_avitopy.npz", X_valid)

del X_train
del X_valid


X_train = sparse.load_npz(DATA_DIR + "X_train_avitopy.npz")
X_valid = sparse.load_npz(DATA_DIR + "X_valid_avitopy.npz")


with ThreadPool(processes=1) as pool:
    Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
    xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
    y_pred = np.mean(pool.map(partial(fit_predict_LGBM, y_train=y_train), xs), axis=0)


y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0]

from sklearn.metrics import mean_squared_error

print('Valid RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(valid['deal_probability'], y_pred))))

# No tweaks. Few inputs.
# => 0.2326
#
#
# Fix exp on pred
#
# => Valid RMSE: 0.2341
#
# Added many fields to text.
#
#
# => 
#
# error    float() argument must be a string or a number, not 'Timestamp'
#
# => Scale and process float (price). propery handle category
#
# * removed date since trouble.
# 
# => X_train: (1428252, 400001) of float32
#
# => 0.2338
#
# => With stemmer
# => 0.2331
#
#
# Back to description, name only. With stemmer
#
# => Valid RMSE: 0.2360
#
#
# Adding cats as text and increase dictionary size
# No activation on last:
#
# 17,000, 170.000. All to text exceptt image_top_1 (discarded)
#
# Valid RMSE: 0.2269
#
# With sigmoid:
# => Valid RMSE: 0.2363
# 
# 
# 170,000, 170,000. All to text with keyword. Added image_top_1 to text.
#
# => BEST 0.2259  three epochs
#
#
# 7 epochs:
#
# => 






 


















