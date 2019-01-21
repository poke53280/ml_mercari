


import spacy

nlp = spacy.load("nb_dep_ud_sm")

doc = nlp(u"Per søker om utenlandsopphold.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
"""c"""

doc = nlp(u"torsk laks tiger ddwqddd.")

for token in doc:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)




nlp_eng = spacy.load("en")


doc_eng = nlp_eng(u"Hello there, I am from Skien.")

doc = nlp_eng(u"Hello there, I am from Skien.")

for token in doc:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
"""c"""

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
"""c"""






for token1 in doc:
    print (token1.voect)
    for token2 in doc:
        print(token1.text, token2.text, token1.similarity(token2))


from gensim.models.wrappers import FastText

import numpy as np
import pandas as pd



df = pd.read_csv(DATA_DIR + 'wiki.no.vec', chunksize = 1000, delimiter = ' ')


