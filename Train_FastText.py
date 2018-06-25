
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
from sklearn.cross_validation import KFold

######################################### CONFIGURATION #######################################

_embeddings_index = {}

################################################################################################

DATA_DIR_PORTABLE = "C:\\avito_data\\"
DATA_DIR_BASEMENT = "C:\\avito_data\\"
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
    # model.summary()
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

    results = np.array((list(map(float, numbers_out))))
    results[::-1].sort()

    return ( results[0], results[1], results[2], results[3], words)

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

def preprocess(df):
    df_desc = preprocess_description(df.description.fillna('NA'), True)
    df_title = preprocess_description(df.title.fillna('NA'), True)

    df['d0'] = df_desc[0]
    df['d1'] = df_desc[1]
    df['d2'] = df_desc[2]
    df['d3'] = df_desc[3]

    df['t0'] = df_title[0]
    df['t1'] = df_title[1]
    df['t2'] = df_title[2]
    df['t3'] = df_title[3]

    df['title'] = df_title[4]

    df['description'] = df_desc[4]

    return df

"""c"""

def load_embedding():

    e = {}

    f = open(EMBEDDING_FILE, encoding="utf8")

    next(f) # skip header

    for o in tqdm(f):
        key, coefs = get_coefs(*o.rstrip().rsplit(' '))
        e[key] = coefs

    f.close()
    return e
"""c"""


<<<<<<< HEAD
_embeddings_index = load_embedding()

training = pd.read_csv(TRAIN_CSV, index_col = "item_id", parse_dates = ["activation_date"])

# training = training[:110]

testing = pd.read_csv(TEST_CSV, index_col = "item_id", parse_dates = ["activation_date"])

# testing = testing[:90]

=======
train = pd.read_csv(TRAIN_CSV, index_col = 0)
# test = pd.read_csv(TEST_CSV, index_col = 0)

train = preprocess(train)
# test = preprocess(test)
>>>>>>> b7c4baa63ca47c4f574e80a69097b38b5db667b9

training = preprocess(training)
testing = preprocess(testing)

training.to_pickle(DATA_DIR + "tr_fast_vec_V1.pkl")
testing.to_pickle(DATA_DIR + "te_fast_vec_V1.pkl")



###
train = training.copy()
test = testing.copy()

labels = train[['deal_probability']].copy()
train = train[['description']].copy()

test = test[['description']].copy()

<<<<<<< HEAD
NFOLDS = 4
=======
# FIT ON TRAIN - !!! ERROR - FITTING ALSO ON VALIDATION
>>>>>>> b7c4baa63ca47c4f574e80a69097b38b5db667b9

ntrain = len (train)
ntest  = len (test)

#kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=121) - out of cuda memory on 2nd fold. 

x_train = train['description'].values
y_train = labels['deal_probability'].values
x_test_const = test['description'].values

oof_train = np.zeros((ntrain,))
oof_test =  np.zeros((ntest,))
oof_test_skf = np.empty((NFOLDS, ntest))

x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=0.10, random_state=23)

#for iLoop, (train_index, test_index) in enumerate(kf):
    
#    print(f"--------------------- Fold {iLoop} ---------------------------")

    #x_tr = x_train[train_index]
    #y_tr = y_train[train_index]
    #x_te = x_train[test_index]
    #y_te = y_train[test_index]
x_test = x_test_const

tokenizer = text.Tokenizer()
maxlen = 60
tokenizer.fit_on_texts(x_tr)
word_index = tokenizer.word_index
nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words, 300))

for word, iEmb in tqdm(word_index.items()):
    if iEmb >= nb_words:
        continue
    
    embedding_vector = _embeddings_index.get(word)

    if embedding_vector is None:
        print(f"Unknown word:'{word}'")
    else:
        embedding_matrix[iEmb] = embedding_vector

x_tr = tokenizer.texts_to_sequences(x_tr)
x_te = tokenizer.texts_to_sequences(x_te)
x_test = tokenizer.texts_to_sequences(x_test)

x_tr = sequence.pad_sequences(x_tr, maxlen=maxlen)
x_te = sequence.pad_sequences(x_te, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = build_model()

history = model.fit(x_tr, y_tr, batch_size = 256, epochs = 2, validation_data = (x_te, y_te), verbose = 1)


x_train = tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

y_p_TEST = model.predict(x_test)
y_p_TEST = y_p_TEST.ravel()

y_p_TRAIN = model.predict(x_train)
y_p_TRAIN = y_p_TRAIN.ravel()


training['y_nn'] = y_p_TRAIN
testing['y_nn'] = y_p_TEST

training.to_pickle(DATA_DIR + 'train_w_nn_2.pkl')
testing.to_pickle(DATA_DIR + 'test_w_nn_2.pkl')


# EXIT




    y_p_X = model.predict(x_te)

    y_p_X = y_p_X.ravel()

    oof_train[test_index] = y_p_X

    y_p_TEST = model.predict(x_test)

    y_p_TEST = y_p_TEST.ravel()
     
    oof_test_skf[iLoop, :] = y_p_TEST

oof_test[:] = oof_test_skf.mean(axis=0)
r0 = oof_train.reshape(-1, 1)
r1 = oof_test.reshape(-1, 1)
    
train['pred_nn'] = r0
test['pred_nn'] = r1



    


rMissingPct = 100.0 * nNotFound/ (nFound + nNotFound)
print(f"text words with embedding: {nFound}. Without: {nNotFound}. Missing {rMissingPct:.1f}%")


<<<<<<< HEAD
=======
history = model.fit(X_train, y_train, batch_size = 128, epochs = 2, validation_data = (X_valid, y_valid), verbose = 1, callbacks = [check_point])
>>>>>>> b7c4baa63ca47c4f574e80a69097b38b5db667b9

prediction_te = model.predict(X_valid)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, prediction_te)))

prediction_Xtest = model.predict(X_test,batch_size = 128, verbose = 1)






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
