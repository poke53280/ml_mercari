
import pandas as pd
from keras.preprocessing import text, sequence
import numpy as np
from tqdm import tqdm
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os

import re
import gc

from keras.layers import Flatten

######################################### CONFIGURATION #######################################

_embeddings_index = {}

################################################################################################

DATA_DIR_PORTABLE = "C:\\avito_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

TRAIN_CSV = DATA_DIR + 'train.csv'
TEST_CSV = DATA_DIR + 'test.csv'



EMBEDDING_FILE = DATA_DIR + 'wiki.ru.vec'

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
"""c"""

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model():
    inp = Input(shape = (maxlen, ))
    emb = Embedding(nb_words, 300, weights = [embedding_matrix],
                    input_length = maxlen, trainable = True)(inp)
    main = SpatialDropout1D(0.2)(emb)
    main = Bidirectional(CuDNNGRU(128,return_sequences = True))(main)
    main = GlobalAveragePooling1D()(main)
    main = Dropout(0.2)(main)
    out = Dense(1, activation = "sigmoid")(main)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error', metrics =[root_mean_squared_error])
    model.summary()
    return model


def build_model_DENSE():
    inp = Input(shape = (maxlen, ))
    emb = Embedding(nb_words, 300, weights = [embedding_matrix],
                    input_length = maxlen, trainable = True)(inp)

    main = Flatten()(emb)

    main = Dense(192)(main)
    main = Dense(64) (main)
    out = Dense(1, activation = "sigmoid")(main)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error', metrics =[root_mean_squared_error])
    model.summary()
    return model



########################################################################
#
#    clean_desc
#
#

def clean_desc(input, isRemoveOOV):

    input = input.lower()
    numbers_out = re.findall(r'\d+', input)
    words = re.sub("\d+", " PARAM ", input)
    words = re.findall(r'[\w]+',words,re.U)

    if (isRemoveOOV):
        words_InV = []

        for word in words:
            if _embeddings_index.get(word) is None:
                # print("Deleting word")
                pass
            else:
                words_InV.append(word)
        
        words = words_InV

    words = " ".join(words)

    numbers_out.append('0')
    numbers_out.append('0')
    numbers_out.append('0')
    numbers_out.append('0')

    return ( numbers_out[0], numbers_out[1], numbers_out[2], numbers_out[3], words)

"""c"""

########################################################################
#
#    preprocess_description
#
#

def preprocess_description(s, isRemoveOOV):
    s_out = s.apply(lambda x: clean_desc(x, isRemoveOOV))
    df = s_out.apply(pd.Series)        
    
    return df

"""c"""


f = open(EMBEDDING_FILE, encoding="utf8")

next(f) # skip header

for o in tqdm(f):
    key, coefs = get_coefs(*o.rstrip().rsplit(' '))
    _embeddings_index[key] = coefs

"""c"""

f.close()


def preprocess(df):
    df_desc = preprocess_description(df.description.fillna('NA'), True)
    df_title = preprocess_description(df.title.fillna('NA'), True)

    #train['d0'] = df_desc[0]
    #train['d1'] = df_desc[1]
    #train['d2'] = df_desc[2]
    #train['d3'] = df_desc[3]

    #train['t0'] = df_title[0]
    #train['t1'] = df_title[1]
    #train['t2'] = df_title[2]
    #train['t3'] = df_title[3]

    df['description'] = df_title[4] + ' ' + df_desc[4]

    return df

"""c"""


train = pd.read_csv(TRAIN_CSV, index_col = 0)
test = pd.read_csv(TEST_CSV, index_col = 0)

train = preprocess(train)
test = preprocess(test)



labels = train[['deal_probability']].copy()
train = train[['description']].copy()


# FIT ON TRAIN

tokenizer = text.Tokenizer()

maxlen = 100  # !!!train['description'].str.len().mean() + 1 * train['description'].str.len().std()

tokenizer.fit_on_texts(list(train['description'].values))

word_index = tokenizer.word_index

nb_words = len(word_index)

embedding_matrix = np.zeros((nb_words, 300))

# Fill embedding matrix with vectors from idx 0 to nb_words

nFound = 0
nNotFound = 0


for word, i in tqdm(word_index.items()):
    if i >= nb_words:
        continue
    
    embedding_vector = _embeddings_index.get(word)

    if embedding_vector is None:
        print(f"Unknown word:'{word}'")
        nNotFound = nNotFound + 1
    else:
        embedding_matrix[i] = embedding_vector
        nFound = nFound + 1

"""c"""

rMissingPct = 100.0 * nNotFound/ (nFound + nNotFound)
print(f"text words with embedding: {nFound}. Without: {nNotFound}. Missing {rMissingPct:.1f}%")


# Preprocess and translate test
# XXXXXXX



test['description'] = test['title'].fillna('NA') + ' ' + test['description'].fillna('NA')

test = test[['description']].copy()

test['description'] = test['description'].astype(str)




X_test = test['description'].values
X_test = tokenizer.texts_to_sequences(X_test)

print('padding')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)





del _embeddings_index

l = list (KFold(n_splits=7, shuffle=True, random_state=42).split(train))

y_pred = np.zeros(len(train))
Y_test = np.zeros(len(test))



X_train, X_valid, y_train, y_valid = train_test_split(train['description'].values, labels['deal_probability'].values, test_size = 0.1, random_state = 23)



print('convert to sequences')
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)

print('padding')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)


model = build_model()

file_path = DATA_DIR + "model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)

history = model.fit(X_train, y_train, batch_size = 256, epochs = 2, validation_data = (X_valid, y_valid), verbose = 1, callbacks = [check_point])

#model.load_weights(file_path)

prediction = model.predict(X_valid)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, prediction)))

test = pd.read_csv(TEST_CSV, index_col = 0)

test['description'] = test['title'].fillna('NA') + ' ' + test['description'].fillna('NA')

test = test[['description']].copy()

test['description'] = test['description'].astype(str)
X_test = test['description'].values
X_test = tokenizer.texts_to_sequences(X_test)

print('padding')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
prediction = model.predict(X_test,batch_size = 128, verbose = 1)

sample_submission = pd.read_csv(DATA_DIR + 'sample_submission.csv', index_col = 0)
submission = sample_submission.copy()
submission['deal_probability'] = prediction
submission.to_csv(DATA_DIR + 'submission.csv')

# no stem, no stop. features: 150,000 GRU 64, wikipedia, batch size 16, mean + 1 std 
# RMSE: 0.233171126161
# LB : 0.2375


# Predict on full train set (should have been oof):

train.to_pickle(DATA_DIR + 'train_w_pred_NN')

train.to_csv(DATA_DIR + 'train_w_pred_NN')
