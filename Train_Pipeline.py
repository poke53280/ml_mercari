

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, make_union, Pipeline

from nltk.corpus import stopwords

import regex as re

from operator import itemgetter

from sklearn.base import BaseEstimator, TransformerMixin


df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

df['DESC'] = pd.util.testing.rands_array(10, 100)

#
# A => float value and scale
# B => Category, to one hot or binary encoded.
#
# DESC => 'char is a word' - to tfidf vectorizer.
#
# D => Is target
#
#




###################################################################################################
#
# FunctionTransformer
#

def test_np_log():
    X = np.arange(10).reshape((5, 2))
 
    # Test that the numpy.log example still works.
    assert_array_equal(
        FunctionTransformer(np.log1p).transform(X),
        np.log1p(X),
    )

"""c"""


def my_func(a):
    print(a.shape)
    return 11

"""c"""

X = np.array([[0, 1], [2, 3]])

tranf2 = FunctionTransformer(my_func)

tranf2.transform(X)


from sklearn.base import BaseEstimator, TransformerMixin

###################################################################################################
#
#    AverageWordLengthExtractor
#

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):

    """Takes in X of strings, returns array of average lengths"""

    def do_something_to(self, X):

        out = []

        for s in X:
            l = _p.split(s)

            nWords = len(l)

            if nWords == 0:
                return 0

            sum = 0

            for x in l:
                sum = sum + len(x)

            out.append(sum/nWords)
        
        return np.matrix(out).T

    def __init__(self):
        _p = re.compile(r'\W+')

    def transform(self, X, y=None):
        return self.do_something_to(X)  # where the actual feature extraction happens

    def fit(self, X, y=None):
        return self  # generally does nothing

"""c"""


from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer, RussianStemmer



###################################################################################################
#
# Pipeline
#

from sklearn.pipeline import Pipeline

   

def x_out_equals(x_a, x_b):
    if x_a.shape != x_b.shape:
        return False

    b_matrix = (x_a != x_b)

    equal = (x_out_0 != x_out_1).nnz == 0

    return equal

"""c"""

x = []

x.append("housing, boats, flying. Driving in cars")
x.append("boating fly drive with car")
x.append("Exile. boats and car. Exciting.")



###################################################################################
#
#   Get_Word_Count_Pipeline
#
#

def Get_Word_Count_Pipeline():

    l3 = []

    l3.append( ('cv', CountVectorizer(ngram_range=(1, 2), analyzer='word')))
    l3.append( ('ss', MaxAbsScaler()         ))

    return Pipeline(l3)
"""c"""


word_count_pipeline = Get_Word_Count_Pipeline()

# Test fit and transform:
x_t = word_count_pipeline.fit_transform(x)

# p3.fit(x, y)
# y_t = p3.predict(x_p)
#
#
# Create average word length extractor
#

l_extract = []

l_extract.append(  ('av', AverageWordLengthExtractor())   )
l_extract.append(  ('ssc', StandardScaler() ) )

av_pipeline = Pipeline(l_extract)

# Test transform:

x_av = av_pipeline.fit_transform(x)


l_stem = []

l_stem.append ( ('ss', SplitAndStemStage() ) )
l_stem.append ( ('cv', CountVectorizer()  ))

stem_pipeline = Pipeline(l_stem)

q = df[9000:9010]

x_stem = stem_pipeline.fit_transform(q.item_description)

x_stem


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

"""c"""

pipe = on_field('item_description', stem_pipeline)

pipe.fit_transform(q)


#
# Got two pipelines - word_count_pipeline and av_pipeline
#
#

from sklearn.pipeline import FeatureUnion

lParallelFeatures = [ ('wc', word_count_pipeline), ('ave', av_pipeline) ]

f = ('feats', FeatureUnion(lParallelFeatures))

lFinalPipeline = []

lFinalPipeline.append(f)

c = ('final', DecisionTreeClassifier())

lFinalPipeline.append(c)

final_pipeline = Pipeline(lFinalPipeline)

"""c"""

final_pipeline.fit(x, y)

final_pipeline.predict(x_p)


########################################################################
#
#  Test: String to list of words transformation
#
#
# FunctionTransformer
#

def sentence_to_word_list(X):
    _p = re.compile(r'\W+')
    x_out = []

    for s in X:
        l = _p.split(s)

        x_out.append(l)
        
    return x_out

"""c"""

t = ('ft', FunctionTransformer(func = sentence_to_word_list))

lp = []

lp.append(t)

p = Pipeline(lp)

ax = np.array(x)

ax = ax.reshape(-1,1)

p.transform(ax)



###################################################################################################
#
#    PlussOneStage
#

class PlussOneStage(BaseEstimator, TransformerMixin):

    def do_something(self, X):

        out = []

        for n in X:
            out.append(n +1 + self._max)
        
        return np.matrix(out).T

    def __init__(self):
        self._max = 0       

    def transform(self, X, y=None):
        return self.do_something(X)  # where the actual feature extraction happens

    def fit(self, X, y=None):

        current_max = 0

        for n in X:
            current_max = np.max([current_max, n])

        self._max = current_max
        return self  # generally does nothing

"""c"""

############################################################################################################
#
# Test pipeline, one for each column  
#
#


from sklearn.pipeline import make_pipeline, make_union, Pipeline
from typing import List, Dict
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from operator import itemgetter

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df[['shipping', 'item_condition_id']]


DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
DATA_DIR_BASEMENT = "D:\\mercari\\"
DATA_DIR = DATA_DIR_PORTABLE


df = pd.read_table(DATA_DIR + "train.tsv");


q = df[:10]
q_test = df[10:13]

q.price.isnull().sum()
q_test.price.isnull().sum()

vectorizer = make_union(
        on_field(['shipping', 'item_condition_id'], PlussOneStage() ),
        n_jobs=1)

p = on_field('item_condition_id', PlussOneStage())

p.fit(q)

X = p.transform(q)

X_test = p.transform(q_test)


X_train = vectorizer.fit_transform(preprocess(q)).astype(np.float32)


X_test = vectorizer.transform(preprocess(q_test)).astype(np.float32)





