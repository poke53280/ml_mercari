

import pandas as pd

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'txt_db.csv')

train.columns= ['1', 'txt', 'target']
train = train.drop(['1'], axis = 1)

from keras.preprocessing.text import text_to_word_sequence

tokenizer = Tokenizer(split=' ')


tokenizer.fit_on_texts(train.txt)

import CountVectorize