
###########################################################

from nltk.stem.snowball import NorwegianStemmer

stemmer = NorwegianStemmer()
stemmer.stem("bilene")

###########################################################

from spacy.lang.nb import Norwegian
nlp = Norwegian()

doc = nlp(u'Dette er en setning.')

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

##########################################################

from gensim.models.wrappers import FastText








import numpy as np
import pandas as pd

DATA_DIR = "C:\\NORDBANK\\"

df = pd.read_csv(DATA_DIR + 'wiki.no.vec', chunksize = 1000, delimiter = ' ')


df_c = pd.DataFrame(df.get_chunk(1500))

df_c = df_c.reset_index()

df_c


# Vanlige ord

DATA_DIR = "C:\\NORDBANK\\"


df_bm = pd.read_csv(DATA_DIR + '\\ordbank_bm\\fullform_bm.txt', skiprows = 25, encoding = 'latin-1', error_bad_lines = False, warn_bad_lines = True, sep = '\t')
df_bm.columns = ['A', 'B', 'C', 'D', 'E', 'F']


df_nn = pd.read_csv(DATA_DIR + '\\ordbank_nn\\fullform_nn.txt', skiprows = 25, encoding = 'latin-1', error_bad_lines = False, warn_bad_lines = True, sep = '\t')
df_nn.columns = ['A', 'B', 'C', 'D', 'E', 'F']


df = pd.concat([df_bm, df_nn], axis = 0)

df.B = df.B.fillna('tom')
df.C = df.C.fillna('tom')

s = set (df.B) | set (df.C)

# Remove and shorter

l = list (s)

l_long = []

for x in l:
    if len(x) > 2:
        l_long.append(x)
"""c"""

s = set(l_long)





