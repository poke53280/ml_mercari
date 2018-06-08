

from nltk.stem.snowball import RussianStemmer, EnglishStemmer, NorwegianStemmer

from sklearn.base import BaseEstimator, TransformerMixin

import regex as re
from nltk.corpus import stopwords


###################################################################################################
#
#    StemmerStage
#
#

class StemmerStage(BaseEstimator, TransformerMixin):

    def do_something_to(self, X):

        N = len (X)

        out = []

        for s in X:
            
            print(f"Processing row {len(out)} of {N}...")
            
            l = self._p.split(s)

            nWords = len(l)

            if nWords == 0:
                return 0

            this_str = ""

            for x in l:

                # print(f"Checking '{x}'")
                if x in self._stopWords:
                    continue

                x = self._stemmer.stem(x)
                
                if len(this_str) == 0:
                    this_str = this_str + x
                else:
                    this_str = this_str + "," + x

            out.append(this_str)
        
        return out

    def __init__(self, l):
        self._p = re.compile(r'\W+')

        assert (l == 'r' or l == 'n' or l == 'e')

        if l == 'r':
            self._stemmer = RussianStemmer()
            self._stopWords = stopwords.words('russian')
        elif l == 'e':
            self._stemmer = EnglishStemmer()
            self._stopWords = stopwords.words('english')
        elif l == 'n':
            self._stemmer = NorwegianStemmer()
            self._stopWords = stopwords.words('norwegian')


    def transform(self, X, y=None):
        return self.do_something_to(X)

    def fit(self, X, y=None):
        return self 

"""c"""
