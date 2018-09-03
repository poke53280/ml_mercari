w
#
# This is an original fork from:
#  https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
#
# by Konstantin Lopuhin and Pawel
# published for mercari.
#
#


# Make work for both at the same time

TRAIN_FILE_AVITO = "C:\\avito_data\\train.csv"
TRAIN_FILE_MERCARI = "C:\\mercari_data\\train.tsv"



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
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

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
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):
            model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        return model.predict(X_test)[:, 0]

def main():
    
    vectorizer = make_union(on_field('name', Tfidf(max_features=100000, token_pattern='\w+')), on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))), on_field(['shipping', 'item_condition_id'], FunctionTransformer(to_records, validate=False), DictVectorizer()))

    y_scaler = StandardScaler()
    
    train = pd.read_table(TRAIN_FILE_MERCARI)

    train = train[train['price'] > 0].reset_index(drop=True)
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train))
    train, valid = train.iloc[train_ids], train.iloc[valid_ids]

    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))

    X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)

    print(f'X_train: {X_train.shape} of {X_train.dtype}')
    del train
    
    X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    
    with ThreadPool(processes=8) as pool:
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        y_pred = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)


    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])

    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['price'], y_pred))))

if __name__ == '__main__':
    main()




