#
# Based on: https://www.kaggle.com/christofhenkel/fasttext-starter-description-only
# by: https://www.kaggle.com/christofhenkel
#
#
# See also:
# https://www.kaggle.com/christofhenkel/using-train-active-for-training-word-embeddings
#
# http://rusvectores.org/en/models/#models
#


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

DATA_DIR_PORTABLE = "C:\\avito_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

EMBEDDING_FILE = DATA_DIR + 'wiki.ru.vec'
TRAIN_CSV = DATA_DIR + 'train.csv'
TEST_CSV = DATA_DIR + 'test.csv'

max_features = 100000

embed_size = 300

train = pd.read_csv(TRAIN_CSV, index_col = 0)


train['description'] = train['title'].fillna('NA') + ' ' + train['description'].fillna('NA')


labels = train[['deal_probability']].copy()
train = train[['description']].copy()

tokenizer = text.Tokenizer(num_words=max_features)
print('fitting tokenizer')

train['description'] =  train['description'].astype(str)


maxlen = train['description'].str.len().mean() + 3 * train['description'].str.len().std()

maxlen = int (maxlen) + 1

print(f"Setting text length to mean + 1 std which is: {maxlen}")


tokenizer.fit_on_texts(list(train['description'].fillna('NA').values))


print('getting embeddings')

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
"""c"""

f = open(EMBEDDING_FILE, encoding="utf8")

next(f) # skip header

embeddings_index = {}

for o in tqdm(f):
    key, coefs = get_coefs(*o.rstrip().rsplit(' '))
    embeddings_index[key] = coefs

"""c"""

f.close()

word_index = tokenizer.word_index


nb_words = min(max_features, len(word_index))   # len word_index: 692156, with title: 752760

embedding_matrix = np.zeros((nb_words, embed_size))  # start randomized (train init)?

# Fill embedding matrix with vectors from idx 0 to nb_words


nFound = 0
nNotFound = 0

for word, i in tqdm(word_index.items()):
    if i >= nb_words: continue
    embedding_vector = embeddings_index.get(word)

    if embedding_vector is None:
        nNotFound = nNotFound + 1
    else:
        embedding_matrix[i] = embedding_vector
        nFound = nFound + 1

"""c"""


print(f"Found: {nFound}. Not found: {nNotFound}")

#
# max_features 100.000. no stem    - about 90% found, 10% missing.
# max_features 300.000. no stem.   - about 60% found, 40% missing.
#
# max_features 300.000. stem and stop word removal:  about 60% found, 40% missing. NO USE
#
# max_features 150.000. no stem    - about 80% found, 20% missing. USING


del embeddings_index


X_train, X_valid, y_train, y_valid = train_test_split(train['description'].values, labels['deal_probability'].values, test_size = 0.1, random_state = 23)


del train



print('convert to sequences')
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)

print('padding')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model():
    inp = Input(shape = (maxlen, ))
    emb = Embedding(nb_words, embed_size, weights = [embedding_matrix],
                    input_length = maxlen, trainable = True)(inp)
    main = SpatialDropout1D(0.3)(emb)
    main = Bidirectional(CuDNNGRU(32,return_sequences = True))(main)
    main = GlobalAveragePooling1D()(main)
    main = Dropout(0.3)(main)
    out = Dense(1, activation = "sigmoid")(main)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error', metrics =[root_mean_squared_error])
    model.summary()
    return model

EPOCHS = 2

model = build_model()
file_path = DATA_DIR + "model7.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)

history = model.fit(X_train, y_train, batch_size = 16, epochs = EPOCHS, validation_data = (X_valid, y_valid), verbose = 1, callbacks = [check_point])

model.load_weights(file_path)

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


