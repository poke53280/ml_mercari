

# Text preprocessing.
# Sequences in general use.
# Keras.
# LSTM. Embeddings


from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

from keras.layers.embeddings import Embedding
from keras.layers import LSTM

import numpy as np

from sklearn.model_selection import train_test_split

tokenizer = Tokenizer()

texts = ["The sun is shining in June!","September is grey.","Life is beautiful in August.","I like it","This and other things?"]
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)
tokenizer.texts_to_sequences(["June is beautiful and I like it!"])


tokenizer.texts_to_matrix(["June is beautiful and I like it!","Like August"])


tokenizer = Tokenizer(char_level=True, oov_token='x')

texts =["abcd-"]
tokenizer.fit_on_texts(texts)

print(tokenizer.word_index)

n = tokenizer.texts_to_sequences(["abaa---aa"])


# String to binary classification, LSTM network.


X = tokenizer.texts_to_matrix(texts)
y = np.array([1.0,0,0,0,0])


vocab_size = len(tokenizer.word_index) + 1

model = Sequential()

model.add(Dense(2, input_dim=vocab_size))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='rmsprop')

X = X.astype(np.float32)

model.fit(X, y, epochs=700, validation_split= 0.2)

from keras.utils.np_utils import np as np

model.predict(X)

X_p = tokenizer.texts_to_matrix(["hello", "shining", "shining June", "Overcast August"])

model.predict(X_p)


model = Sequential()

model.add(Embedding(2,2, input_length = 7))

model.compile('rmsprop', 'mse')
model.predict(np.array([[0, 1, 0, 1, 1, 0, 0]]))

######################################################################################

from keras.datasets import imdb

top_words = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

type (X_train)


y_train

X_Train


#######################################################################################################
#
#   How to use Different Batch Sizes when Training and Predicting with LSTMs

# https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/


from pandas import concat
from pandas import DataFrame

length = 10
sequence = [i/float(length) for i in range(length)]

df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
print(df)






##################################################################################################

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

from keras.layers import LSTM

import numpy as np

from sklearn.model_selection import train_test_split



def prepare_sequences(X, y, window_length):
    windows = []
    windows_y = []
    for i, sequence in enumerate(X):
        len_seq = len(sequence)
        for window_start in range(0, len_seq - window_length + 1):
            window_end = window_start + window_length
            window = sequence[window_start:window_end]
            windows.append(window)
            windows_y.append(y[i])
    return np.array(windows), np.array(windows_y)

"""c"""

N_train = 1000
from numpy.random import choice
one_indexes = choice(a=N_train, size=N_train // 2, replace=False)


X = np.zeros((1200, 20), dtype =np.float)

X[one_indexes, 0] = 1


X = X.astype(np.float32)

y = X[:,0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 200/1200, random_state = 111)

#X_train, y_train = prepare_sequences(X_train, y_train, 10)
#X_test, y_test = prepare_sequences(X_test, y_test, 10)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)



model = Sequential()

model.add(LSTM(10, batch_input_shape=(1, 1, 1), return_sequences=False, stateful=True))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print('Train...')
for epoch in range(15):
    mean_tr_acc = []
    mean_tr_loss = []
    for i in range(len(X_train)):
        y_true = y_train[i]
        for j in range(max_len):
            tr_loss, tr_acc = model.train_on_batch(np.expand_dims(np.expand_dims(X_train[i][j], axis=1), axis=1),
                                                   np.array([y_true]))
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)
        model.reset_states()

    print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
    print('loss training = {}'.format(np.mean(mean_tr_loss)))
    print('___________________________________')

    mean_te_acc = []
    mean_te_loss = []
    for i in range(len(X_test)):
        for j in range(max_len):
            te_loss, te_acc = model.test_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=1), axis=1),
                                                  y_test[i])
            mean_te_acc.append(te_acc)
            mean_te_loss.append(te_loss)
        model.reset_states()

        for j in range(max_len):
            y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=1), axis=1))
        model.reset_states()

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    print('loss testing = {}'.format(np.mean(mean_te_loss)))
    print('___________________________________')



#max_len  = 10
#model.add(LSTM(10, input_shape=(max_len, 1), return_sequences=False, stateful=False))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size=1
shuffle=False 

model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_test, y_test), shuffle=False)






from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence

# define the document

text = 'The quick brown fox jumped over the lazy dog.'

text = 'a a x x x a x x a a x x a a a a'

# estimate the size of the vocabulary

words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

# integer encode the document
result = one_hot(text, round(vocab_size*1.3))
print(result)




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
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs

def get_model(nWords, input_length, eMatrix, nEmbDim, nClasses):
    
    num_filters = 64
    weight_decay = 1e-4
    

    model = Sequential()

    model.add(Embedding(nWords, nEmbDim, weights=[eMatrix], input_length=input_length, trainable=False))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(nClasses, activation='sigmoid'))  #multi-label (k-hot encoding)

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

def mse_function(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean()
    return mse

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



def keras_CV(model, X, y, splits, nEpochs, nBatchSize, type_str):
    
    kf = KFold(n_splits = splits, random_state = 122)
    
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

        hist = model.fit(train_X, train_y, batch_size=nBatchSize, epochs=nEpochs, callbacks=callbacks_list, validation_data=(valid_X, valid_y), shuffle=True, verbose=2)

        o = 0

        y_pred = model.predict_proba(valid_X)

        if type_str == "MULTI_LABEL":
            o = error_function(valid_y, y_pred)
        elif type_str == "REGRESSION":
            o = mse_function(valid_y, y_pred)
        else:
            print("ERROR")

        print("Accuracy = " + str(o))

        lcAccuracy.append(o)
        nFold = nFold + 1

   

    d = {}

    d['score'] = np.array(lcAccuracy).mean()
    d['std'] = np.array(lcAccuracy).std()

    return d

"""c"""

def mymain():

    """CONFIGURATION"""

    # consts

    DATA_DIR = "C:\\Users\\T149900\\ml_mercari\\"
    WORD_COUNT_MEAN_PLUSS_STD = 1
    WORD_COUNT_MEAN_THIRD = 2

    num_words = 100000

    batch_size = 256
    num_epochs = 8

    num_splits = 5


    embed_dim = 300 


    word_count_strategy = WORD_COUNT_MEAN_PLUSS_STD
    word_database = "toxic\\wiki.simple.vec"


# CV accuracy is 0.987488975854 +/- 0.00889961690531
# CV accuracy is 0.988607054794 +/- 0.00992068287633



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

    if word_count_strategy == WORD_COUNT_MEAN_PLUSS_STD:
        max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)
    elif word_count_strategy == WORD_COUNT_MEAN_THIRD:
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

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, lower=True, char_level=False)

    tokenizer.fit_on_texts(processed_docs_train)  #non-leaky

    word_index = tokenizer.word_index

    print("dictionary size: ", len(word_index))

    word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)

    word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)

    embedding_matrix = get_embedding_matrix(embeddings_index, num_words)

    model = get_model(num_words, max_seq_len, embedding_matrix, embed_dim, num_classes)

    d = keras_CV(model, word_seq_train, y_train, num_splits, num_epochs, batch_size)

    print("CV accuracy is " + str(d['score']) + " +/- " + str(d['std']))


#   Gave LB 0.9542
#   epoch 5: acc 0.9829 val_acc 0.9793


