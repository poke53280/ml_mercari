


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



def keras_CV(model, X, y, splits, nEpochs, nBatchSize):
    
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

        y_pred = model.predict_proba(valid_X)
        
        o = error_function(valid_y, y_pred)

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


### Gave LB 0.9542
# epoch 5: acc 0.9829 val_acc 0.9793

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################


#----------------------------------------------------------------------------------------
#     AAAA           A        AAAAAAAAAAAAAAAA            A AA     AAAAAAAA
#   BBB        BBBBBBBBB  BBB                         BBB  B   BBBBBB
# C     C               CCC          C               CCCC             CCCCC       CC


import numpy as np

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



def get_model_2(nWords, input_length, eMatrix, nEmbDim, nClasses):
    
    num_filters = 64
    weight_decay = 1e-4
    
    model = Sequential()

    model.add(Embedding(nWords, nEmbDim, weights=[eMatrix], input_length=input_length, trainable=False))
    model.add(Conv1D(num_filters, 3, activation='relu', padding='same'))  #todo: 3 - adjust
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 3, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(nClasses, activation='sigmoid'))  #multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

"""c"""

def get_embedding_matrix_X(num_dim) :

    embedding_matrix = np.zeros((2**num_dim, num_dim))

    for idx in range(0, 2**num_dim):

        word_vector = []

        for d in range(0,num_dim):
            nDimensionValue = 0

            exp = 2 ** d

            if idx & exp != 0:
                nDimensionValue = 1

            word_vector.append(nDimensionValue)

        embedding_matrix[idx] = word_vector

    return embedding_matrix
    
"""c"""    

def process_lines(lA, lB, lC):
    assert (len(lA) == len(lB))
    assert (len(lA) == len(lC))

    z = zip(lA, lB, lC)

    i_out = []

    for i, val in enumerate(z):
        idx = val[0] * 1 + val[1] * 2 + val[2] * 4
        i_out.append(idx)

    return i_out

"""c""" 

def create_timeline(seq_len):
    a = np.random.choice(2, seq_len).astype(np.float32)
    b = np.random.choice(2, seq_len).astype(np.float32)
    c = np.random.choice(2, seq_len).astype(np.float32)

    return a,b,c

"""c"""

def create_target(a, b, c):
    assert (len(a) == len(b))
    assert (len(a) == len(c))
    
    a_sum = a.sum()
    b_sum = b.sum()
    c_sum = c.sum()

    all_90 = len(a) * 0.9
    all_60 = len(a) * 0.6

    y = np.zeros((7 ), dtype=np.float32)

    if a_sum > c_sum:
        y[0] = 1.0

    if a_sum > b_sum:
        y[1] = 1.0

    if a_sum + b_sum > b_sum + c_sum:
        y[2] = 1.0

    if a_sum + b_sum >  all_60:
        y[3] = 1.0

    if a_sum + c_sum > all_90:
        y[4] = 1.0

    if c_sum >  all_60:
        y[5] = 1.0

    if c_sum + b_sum > all_90:
        y[6] = 1.0

    return y

"""c"""



def create_dataset(num_elements, seq_len):
    p = []
    t = []

    for i in range(0, num_elements):
        a, b, c = create_timeline(seq_len)
        z = zip(a,b,c)
        p.append(z)

        y = create_target(a, b, c)
        t.append(y)

    return p, t

"""c"""

def create_Y(t):
    y = np.zeros((len(t), 7))

    for i, x in enumerate(t):
         y[i] = x

    return y

"""c"""



def create_idx_runs(p, seq_len):
    num_elements = len (p)

    p_i = np.zeros((num_elements, seq_len))

    for i, p_this in enumerate(p):

        a,b,c = zip (*p_this)

        data = process_lines (a, b, c)
        
        p_i[i] = data

    return p_i

"""c"""


word_length = 9 * 1

d = get_embedding_matrix_X(3)

df, t = create_dataset(20, word_length)

X = create_idx_runs(df, word_length)
X = X.astype(np.float32)

y = create_Y(t)
y = y.astype(np.float32)


m = get_model_2(2**3, word_length, d, 3, 7)

print ("X.shape = " + str(X.shape))
print ("y.shape = " + str(y.shape))

res = keras_CV(m, X, y, 10, 8, 16)

print("CV accuracy is " + str(res['score']) + " +/- " + str(res['std']))

"""c"""

# Got X and y and embedding matrix







