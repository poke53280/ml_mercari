


# Diag:
# Doc : 

# Sparse in time

# Day shift 143
# Age 45
 

# DOC   ---------------------------------qqqq-----------qqq-----------------------ff 
# DIA   ---------------------------------yyyy-----------bbb-----------------------yy 

# DOC is categorical
# DIA is categorical

# Two categorical values, begin-end.

# Create embedding?

# ------Doc333---Dia323----------------Dia339-----------Doc933-----Dia900----- => 90

#
#
# ~ 10,000 dia
# ~ 10,000 wrk
# ~ 10,000 doc
#
#






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



def get_model_regression(nWords, input_length, eMatrix, nEmbDim, nClasses):
    
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

    model.add(Dense(1, activation='relu'))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='mean_squared_error', optimizer=adam)
  
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

def process_lines(z):


    i_out = np.zeros(len(z[0]), 1, dtype = np.float32)

    for t in z:
        sum = 0
        for i, x in enumerate(t):
            sum = sum + x * (2**i)

        i_out.append(sum)

    return i_out

"""c""" 


def create_timeline(seq_len):
    q = np.random.choice(2, seq_len).astype(np.float32)
   
    return q

"""c"""

def create_target_regression(l):


    n = l[0].sum() + l[1].sum() + l[2].sum()

    i = np.random.choice([1,2,3], 1)

    if i == 1:
        n = n + l[2].std()
    elif i == 2:
        n = n + l[1].std()
    elif i == 3:
        n = n + l[0].std()

    if l[0].sum() > l[1].sum():
        n = n * 1.1 + l[2].mean()

    if l[1].sum() > l[2].sum():
        n = n * 1.2 + l[1].mean()

    n = n / len (l[0])

    return n


"""c"""



def create_dataset(num_elements, seq_len):
    p = []
    t = []

    for i in range(0, num_elements):
       
        a = create_timeline(seq_len)
        b = create_timeline(seq_len)
        c = create_timeline(seq_len)

        l = []

        l.append(a)
        l.append(b)
        l.append(c)

        p.append(l)

        y = create_target_regression(l)
        t.append(y)

    return p, t

"""c"""


def create_idx_runs(p, seq_len):
    num_elements = len (p)

    p_i = np.zeros((num_elements, seq_len))

    for i, p_this in enumerate(p):

        data = process_lines (p_this)
        
        p_i[i] = data

    return p_i

"""c"""


word_length = 3 * 365

d = get_embedding_matrix_X(3)

df, t = create_dataset(4400, word_length)

X = create_idx_runs(df, word_length)
X = X.astype(np.float32)

y = np.array(t)
y = y.reshape(len(t), 1)

y = y.astype(np.float32)

m = get_model_regression(2**3, word_length, d, 3, 7)

print ("X.shape = " + str(X.shape))
print ("y.shape = " + str(y.shape))

seed = 7
np.random.seed(seed)


res = keras_CV(m, X, y, 10, 8, 32, "REGRESSION")

print("CV accuracy is " + str(res['score']) + " +/- " + str(res['std']))

"""c"""


y_p = m.predict(X)

# Got X and y and embedding matrix





import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import general.dataset as data

tokenizer = Tokenizer(char_level=True, oov_token='x')

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/




l = data.dataset_get_basic_sequence()

y = data.dataset_create_y_large_ac_small_b_regression(l)

L = len (l[0])

# Target:
tokenizer.fit_on_texts(l)

n = tokenizer.texts_to_sequences(l)

X = np.array(n, dtype= np.float32)

np.random.seed(137)

splits = 3

kf = KFold(n_splits = splits)
    
nSplits = kf.get_n_splits(X)

nFold = 0

cm_l = np.zeros((2,2), dtype = np.float32)

lSplit = list (kf.split(X))


for index, item in enumerate(lSplit):

    train_index = lSplit[index][0]
    valid_index = lSplit[index][1]

    print ("FOLD# " + str(index))

    X_train = X[train_index]  
    y_train = y[train_index]

    X_test = X[valid_index]
    y_test = y[valid_index]


    model = Sequential()
    #model.add(Embedding(6, 6, input_length= L ))

    #model.add(LSTM(512, return_sequences=True))
    model.add(Dense(10, input_shape = (1,9)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


    #model.add(Dense(L, input_dim=L, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    #model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=18)

    y_p = model.predict(X_test)

    y_p = (y_p > 0.5)

    y_p = y_p.astype(float)
    
    cm = confusion_matrix(y_test, y_p)

    cm_l = cm_l + cm

    nFold = nFold + 1

"""c"""







