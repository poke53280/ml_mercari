





# * Run mercari solution with mercari dataset. 
# * Make mercari solution run with job dataset. Todo: How to incorporate other fields + see strength alone
#
#
#
# * Make BERT run. Compare performance with the approach below. Consider datasize. Consider transfer learning.
# * Make BERT run with Norwegian dataset.
# * Make BERT run on job dataset.
#

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf

from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer
from operator import itemgetter
from typing import List, Dict
from sklearn.feature_extraction import DictVectorizer


f : FunctionTransformer = FunctionTransformer(func = np.log1p, validate = False)

X = [[3,4,11], [9,9,9]]

f.fit(X)
f.transform(X)

f.get_params()




def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]


def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

y_scaler = StandardScaler()

data = pd.read_table('C:\\Users\\T149900\\Downloads\\mercari.tsv')

cv = KFold(n_splits=20, shuffle=True, random_state=42)

train_ids, valid_ids = next(cv.split(data))

train = data.iloc[train_ids]
valid = data.iloc[valid_ids]

del data

y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))

train = preprocess(train)

name_tfidf = Tfidf(max_features=100000, token_pattern='\w+')

name_p : Pipeline = make_pipeline(FunctionTransformer(itemgetter('name'), validate=False), name_tfidf)

text_tfidf = Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))

text_p = make_pipeline(FunctionTransformer(itemgetter('text'), validate=False), text_tfidf)

shipping_p = make_pipeline(FunctionTransformer(itemgetter('shipping'), validate=False), FunctionTransformer(to_records, validate=False), DictVectorizer())

condition_p = make_pipeline(FunctionTransformer(itemgetter('item_condition_id'), validate=False), FunctionTransformer(to_records, validate=False), DictVectorizer())

vectorizer = make_union(name_p, text_p, shipping_p, condition_p)


X_train = vectorizer.fit_transform(train).astype(np.float32)
