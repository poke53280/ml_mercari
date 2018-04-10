

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs


sns.set_style("whitegrid")
np.random.seed(0)

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\toxic\\"

MAX_NB_WORDS = 10000


print('loading word embeddings...')

embeddings_index = {}

f = codecs.open(DATA_DIR_PORTABLE + 'wiki.simple.vec', encoding='utf-8')

for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

"""c"""

print('found %s word vectors' % len(embeddings_index))

