


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
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs

def get_model(num_words, input_length):
    model = Sequential()

    model.add(Embedding(num_words, embed_dim, weights=[embedding_matrix], input_length=input_length, trainable=False))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

"""c"""

def get_word_embeddings(filename):
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(DATA_DIR + word_database, encoding="utf8")
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))

    return embeddings_index

"""c"""


def get_embedding_matrix(embeddings_index, num_max_words):
    print('preparing embedding matrix...')

    words_not_found = []
    words_not_found_index = []

    nb_words = min(num_max_words, len(word_index))

    embedding_matrix = np.zeros((nb_words, embed_dim))

    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
            words_not_found_index.append(i)

    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return embedding_matrix


"""c"""


def dist_words(w1, w2, e):
    a = (e[w1] + 1) * 0.5
    b = (e[w2] + 1) * 0.5

    o = TXTP_rmsle(a, b)

    print("Distance " + w1 + ", " + w2 +": " + str(o))


"""c"""

def error_function(y_true, y_pred):
    col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    sum = 0

    for i, j in enumerate(col):
        o = roc_auc_score(y_true[:,i], y_pred[:,i])
        sum = sum + o

    return sum/len(col)

"""c"""

def error_function_digitized(y_true, y_pred):
    col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    sum = 0

    for i, j in enumerate(col):
        y_p = np.digitize(y_pred[:,i], [0.5])
        o = roc_auc_score(y_true[:,i], y_p)
        sum = sum + o

    return sum/len(col)

"""c"""



def keras_CV(model, X, y, splits, nEpochs):
    
    kf = KFold(n_splits = splits, random_state = 133)
    
    nSplits = kf.get_n_splits(X)

    nFold = 0

    lcAccuracy = []

    for train_index, valid_index in kf.split(X):

        print ("FOLD# " + str(nFold))

        train_X = X[train_index]  
        train_y = y[train_index]

        valid_X = X[valid_index]
        valid_y = y[valid_index]

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
        callbacks_list = [early_stopping]

        hist = model.fit(train_X, train_y, batch_size=batch_size, epochs=nEpochs, callbacks=callbacks_list, validation_data=(valid_X, valid_y), shuffle=True, verbose=2)

        y_pred = model.predict_proba(valid_X)
        
        o = error_function(valid_y, y_pred)

        print("Accuracy = " + str(o))

        lcAccuracy.append(o)
        nFold = nFold + 1

    print("CV accuracy is " + str(np.array(lcAccuracy).mean()) + "+/-" + str(np.array(lcAccuracy).std()))

    return np.array(lcAccuracy).mean()


"""CONFIGURATION"""

DATA_DIR = "C:\\Users\\T149900\\ml_mercari\\"

MAX_NB_WORDS = 1000

batch_size = 256
num_epochs = 2

num_splits = 10

num_filters = 64
embed_dim = 300 
weight_decay = 1e-4

# 1: mean + std
# 2: mean/3.0

word_count_strategy = 2
word_database = "toxic\\wiki.simple.vec"

sns.set_style("whitegrid")
np.random.seed(0)

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])


embeddings_index = get_word_embeddings(DATA_DIR + word_database)

train_df = pd.read_csv( DATA_DIR + "toxic\\train.csv")

print("num train: ", train_df.shape[0])

label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train_df[label_names].values

print (y_train.shape)

train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))


max_seq_len = 0

if word_count_strategy == 1:
    max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)
elif word_count_strategy == 2:
    max_seq_len = np.round(train_df['doc_len'].mean()/3.0).astype(int)

assert(max_seq_len > 0)

raw_docs_train = train_df['comment_text'].tolist()

num_classes = len(label_names)


print("pre-processing train data...")
processed_docs_train = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))
"""end for"""

tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)

tokenizer.fit_on_texts(processed_docs_train)  #non-leaky

word_index = tokenizer.word_index

print("dictionary size: ", len(word_index))

word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)

word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)


embedding_matrix = get_embedding_matrix(embeddings_index, MAX_NB_WORDS)

model = get_model(MAX_NB_WORDS, max_seq_len)

oKeras = keras_CV(model, word_seq_train, y_train, num_splits, num_epochs)

print("Keras CNN accuracy: " + str(oKeras))



### Gave LB 0.9542
# epoch 5: acc 0.9829 val_acc 0.9793

