


df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

df['DESC'] = pd.util.testing.rands_array(10, 100)

#
# A => float value and scale
# B => Category, to one hot or binary encoded.
#
# DESC => 'char is a word' - to tfidf vectorizer.
#
# D => iS tARGET
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


import numpy as np
from sklearn.preprocessing import FunctionTransformer


def my_func(a):
    return 11

"""c"""

transformer = FunctionTransformer(np.log1p)

X = np.array([[0, 1], [2, 3]])

tranf2 = FunctionTransformer(my_func)

transformer.transform(X)

tranf2.transform(X)


###################################################################################################




