


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

from sklearn.metrics import mean_squared_error


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import concatenate

from keras import optimizers

import gc


import general.dataset as data

def getX(num_tokens, num_cols, num_rows):
    return np.random.random_integers(0, num_tokens -1, (num_rows, num_cols))


def getY(X):

    y = []

    for row in X:
        c = np.bincount(row)

        max = len(c)

        tok_a = int (max * 0.24)
        tok_b = int (max * 0.27)
        tok_c = int (max * 0.91)
        tok_d = int (max * 0.13)

        # score = (c[0] < c[1]) * .2 * c[2] + (c[max-1] > c[max-2]) * .5 * c[4] + (c[3] == c[5]) * 1.1 * np.log (1 + c[3]) + 1 * c[tok_c]

        #if (c[tok_a] < c[tok_b]):
        #    
        #else:
        score = (c[3] < c[1]) * .2 * c[2] + (c[max-4] > c[max-2]) * .7 * c[6] + (c[1] == c[5]) * 1.9 * np.log (4 + c[3]) + 1 * c[tok_b]


        y.append(score)

    y = np.array(y, dtype = np.float32)
    #y = y / y.max()

    return y

"""c"""


nCols = 528       

nRows = 350000

num_tokens = 9000

num_epochs = 6

X = getX(num_tokens, nCols, nRows)

y = getY(X)

print (X.shape)

vocab_size = X.max() - X.min() + 1

assert (vocab_size == num_tokens)

np.random.seed(137)

splits = 10

kf = KFold(n_splits = splits)
    
nSplits = kf.get_n_splits(X)

nFold = 0

lSplit = list (kf.split(X))

l_RMS = []

train_index = lSplit[0][0]
valid_index = lSplit[0][1]

X_train = X[train_index]  
y_train = y[train_index]

X_test = X[valid_index]
y_test = y[valid_index]

input_layer_0 = Input(shape=(nCols,), name = "input_0")
embedding_layer_0 = Embedding(vocab_size, 8, name = "Emb_0")(input_layer_0)

input_layer_1 = Input(shape=(nCols,), name = "input_1")
embedding_layer_1 = Embedding(vocab_size, 8, name = "Emb_1")(input_layer_1)

input_layer_2 = Input(shape=(nCols,), name = "input_2")
embedding_layer_2 = Embedding(vocab_size, 8, name = "Emb_2")(input_layer_2)

input_layer_3 = Input(shape=(nCols,), name = "input_3")
embedding_layer_3 = Embedding(vocab_size, 8, name = "Emb_3")(input_layer_3)

c_layer = concatenate([embedding_layer_0, embedding_layer_1, embedding_layer_2, embedding_layer_3])

flatten_0 = Flatten() (c_layer)

deep_0 = Dense(64, activation='linear') (flatten_0)

deep_1 = Dense(64, activation='linear') (deep_0)

deep_2 = Dense(16, activation='linear') (deep_1)

out = Dense(1)(deep_2)

lcInput = [input_layer_0, input_layer_1, input_layer_2, input_layer_3]

m = Model(inputs= lcInput, outputs=out)

print (m.summary())

opt = optimizers.Adam()

m.compile(loss='mse', optimizer= opt, metrics=['mse'])

lcTrainData = [X_train, X_train, X_train, X_train]

lcTestData = [X_test, X_test, X_test, X_test]

m.fit(lcTrainData, [y_train], validation_data=(lcTestData, [y_test]), epochs=num_epochs, batch_size=2048, verbose=1)

y_p = m.predict([X_test, X_test, X_test, X_test])

error = mean_squared_error(y_test, y_p)


print(f"RMS = {error:.5f}")

#
# confer with embedding approach
# http://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/11/27/Understanding-Convolutions-In-Text/
#
#
#
# Todo: Validation.
# https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
#
# + early stopping et.c.
# + mini batch
#
# + expand dictionary from 'a-z+ and digits (37) to  about 30,000, 6,000 and 50
# + set up separate data sources for three inputs.




