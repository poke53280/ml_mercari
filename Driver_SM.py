
#
# See: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
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

def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                  # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])


def generate_sequence(nRows, nCols, nTokens, nRunsPerRow):
    
    L = nCols

    row = np.zeros((nRows, L), dtype = int)

    for iRow in range(0, nRows):

        nRunsThisSequence = np.random.randint(1, nRunsPerRow +1 )

        for iRun in range(0, nRunsThisSequence):

            # print (iRun)
            set_value = np.random.randint(nTokens, size=1)[0]

            offset = np.random.randint(L, size=1)[0]
            # print (offset)

            length = np.random.randint(1, (L+1)/3)          # 1.. L/3

            # print(f"offset = {offset}, length = {length}")

            row[iRow, offset:offset + length] = set_value

    return row

"""c"""


def get_value_score(value):
    return 1 + np.sin(value/100)

def get_y_row (X):

    y = []

    for row in X:

        t = rle(row)

        score = 0
        for run_length, start, value in zip(*t):
            if run_length < 3:
                continue

            if value == 0:
                continue

            score = score + run_length * get_value_score(value)
        y.append(score)

    y = np.array(y)
    return y

"""c"""


nCols = 1028       

nRows = 950000

num_tokens = 5000

num_epochs = 6

X = generate_sequence(nRows, nCols, num_tokens, 4)

y = get_y_row(X)

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

x = Input(shape=(nCols,), name = "input_0")
embedding_layer_0 = Embedding(vocab_size, 32, name = "Emb_0")(x)

x = Input(shape=(nCols,), name = "input_1")
embedding_layer_1 = Embedding(vocab_size, 16, name = "Emb_1")(x)

x = Input(shape=(nCols,), name = "input_2")
embedding_layer_2 = Embedding(vocab_size, 8, name = "Emb_2")(x)

x = Input(shape=(nCols,), name = "input_3")
embedding_layer_3 = Embedding(vocab_size, 8, name = "Emb_3")(x)

x = concatenate([embedding_layer_0, embedding_layer_1, embedding_layer_2, embedding_layer_3])

x = Flatten() (x)

x = Dense(128, activation='linear') (x)

deep_1 = Dense(64, activation='linear') (x)

deep_2 = Dense(64, activation='linear') (x)

out = Dense(1)(x)

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




