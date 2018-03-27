

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import time


np.random.seed(7)

def process(top_words, max_review_length, nUnits, nEpochs, nBatchSize, embedding_vector_length):    

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


    model = Sequential()

    # The first layer is the Embedded layer that uses 32 length vectors to represent each word.

    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))


    # The next layer is the LSTM layer with 100 memory units (smart neurons)

    model.add(LSTM(nUnits))

    # Finally, because this is a classification problem we use a Dense output layer with a single
    # neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes
    # (good and bad) in the problem.

    model.add(Dense(1, activation='sigmoid'))


    # Regression alternative?
    # model.add(Dense(1, kernel_initializer='normal'))


    # Because it is a binary classification problem, log loss is used as the
    # loss function (binary_crossentropy in Keras). The efficient ADAM optimization
    # algorithm is used. The model is fit for only 2 epochs because it quickly overfits the problem.
    # A large batch size of 64 reviews is used to space out weight updates.


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())


    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nEpochs, batch_size=nBatchSize)

    # Final evaluation of the model

    scores = model.evaluate(X_test, y_test, verbose=0)

    fAccuracy = scores[1]*100

    print(f"Accuracy: {fAccuracy:.2f}")

    print(f"Epochs {nEpochs} batch_size {nBatchSize}")
    
    print(f"LSTM units: {nUnits}")

    print(f"Review length: {max_review_length}")
    print(f"Top words {top_words}")
    print(f"Embedding vector length: {embedding_vector_length}")
    
    print("==> Accuracy: %.2f%%" % (scores[1]*100))

    return fAccuracy

"""c"""




fAccuracy = process(5000, 500, 100, 1, 64, 32)


fAccuracy2 = process(15000, 500, 200, 1, 64, 64)

# load the dataset but only keep the top n words, zero the rest

ai = range(10)

for i in ai:

    top_wordsXXX = int (np.random.choice([5000, 10000]))
    max_review_lengthXXX = int (np.random.choice([500, 200, 800]))
    nUnitsXXX = int (np.random.choice([100, 200]))
    nEpochsXXX = int (np.random.choice([1, 3, 7]))
    nBatchSizeXXX = int (np.random.choice([64, 32, 128]))
    embedding_vector_lengthXXX = int(np.random.choice([32, 16, 64]))

    print(f"Run: {i}...")

    t0 = time.time()

    fAccuracy = process(top_wordsXXX, max_review_lengthXXX, nUnitsXXX, nEpochsXXX, nBatchSizeXXX, embedding_vector_lengthXXX)

    t1 = time.time()

    dT = t1 - t0

    print(f"dT = {dT:.1f}s")


"""c"""



import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


tokenizer = Tokenizer(char_level=True, oov_token='x')

texts =["abcd-"]
tokenizer.fit_on_texts(texts)

print(tokenizer.word_index)

l = []
l.append("abaa---aa")
l.append("ab--bbbaa")
l.append("aa-bbbbaa")
l.append("ba-a-aaaa")
l.append("bbbbbaaab")
l.append("ab--bbbbb")
l.append("bb-bbbbaa")
l.append("aa-bbbbaa")
l.append("aba-bbbba")
l.append("bbaabbbaa")
l.append("aabbbabba")
l.append("baa-bbbaa")
l.append("aba-bbbaa")
l.append("ba-b--aba")
l.append("a-babbbaa")
l.append("abbbbaaab")
l.append("aab-aaaba")
l.append("bbbbaaabb")
l.append("ab--bbabb")
l.append("bb-ababaa")
l.append("aa-babbaa")
l.append("abaab-bba")
l.append("bbaabbbaa")
l.append("aabqbabba")
l.append("baa-abbaa")
l.append("baba-bbaa")
l.append("ab-a--aab")
l.append("ab--bbbaa")
l.append("aacbebbaa")
l.append("ba-a-aaaa")
l.append("bbabbaaab")
l.append("ab-bbgbbb")
l.append("bb-bbbbaa")
l.append("aab-bbbaa")
l.append("aba-bbbba")
l.append("bbaabbbaa")
l.append("aabbbabba")
l.append("baa-bbbaa")
l.append("-aaabbbaa")
l.append("ba-d--aba")
l.append("a-babbbaa")
l.append("abbbcbaaa")
l.append("aab-aaaba")
l.append("bbbbababa")
l.append("ab--bbabb")
l.append("bb-ababaa")
l.append("aa-babbaa")
l.append("ababa-bba")
l.append("bbaaabbaa")
l.append("aabcbbaba")
l.append("baaa-baaa")
l.append("babbba-aa")
l.append("abaa---aa")
l.append("ab--bbbaa")
l.append("aa-bbbbaa")
l.append("ba-a-aaaa")
l.append("bbbbbaaab")
l.append("ab--bbbbb")
l.append("bb-bbbbaa")
l.append("aa-bbbbaa")
l.append("aba-bbbba")
l.append("bbaabbbaa")
l.append("aabbbabba")
l.append("baa-bbbaa")
l.append("aba-bbbaa")
l.append("ba-b--aba")
l.append("a-babbbaa")
l.append("abbbbaaab")
l.append("aab-aaaba")
l.append("bbbbaaabb")
l.append("ab--bbabb")
l.append("bb-ababaa")
l.append("aa-babbaa")
l.append("abaab-bba")
l.append("bbaabbbaa")
l.append("baabqbabb")
l.append("abaa-abba")
l.append("fdfbaba-b")
l.append("fab-a--aa")
l.append("adfab--bb")
l.append("aacadfbeb")
l.append("ba-dfa-aa")
l.append("bbabbaaab")
l.append("ab-bebgbb")
l.append("bb-bbbadf")
l.append("aab-bbasd")
l.append("aba-bbbba")
l.append("basdfbaab")
l.append("aasdfabbb")
l.append("badsfaa-b")
l.append("dfa-aaabb")
l.append("badsfa-d-")
l.append("a-adsfbab")
l.append("dafabbbcb")
l.append("fdafaab-a")
l.append("bbadfbbab")
l.append("ab--adfbb")
l.append("bb-aasdfb")
l.append("aa-asdfba")
l.append("ababa-baa")
l.append("bbaaaafab")
l.append("aabcbbaaa")
l.append("baaa-baaa")
l.append("babbaa-aa")

l.append("abaa---aa")
l.append("ab--bxbaa")
l.append("aa-bbbbaa")
l.append("ba-a-aaaa")
l.append("bbbbbaaab")
l.append("ab--bbbbb")
l.append("bb-bbbbaa")
l.append("aa-bbbbaa")
l.append("aba-bbbba")
l.append("bbaabbbaa")
l.append("aabbbabba")
l.append("baa-bbbaa")
l.append("aba-bbbaa")
l.append("ba-b--aba")
l.append("a-babbbaa")
l.append("abbbbaaab")
l.append("aab-aaaba")
l.append("bbbbaaabb")
l.append("ab--bbabb")
l.append("bb-ababaa")
l.append("aa-babbaa")
l.append("abaab-bba")
l.append("bbaabbbaa")
l.append("aabqbxbba")
l.append("baa-abbaa")
l.append("baba-bbaa")
l.append("ab-a--aab")
l.append("ab--bbbaa")
l.append("aacbecbaa")
l.append("ba-a-aaaa")
l.append("bbabbaaab")
l.append("ab-bbdbbb")
l.append("bb-bbbbaa")
l.append("aab-bbbaa")
l.append("aba-bbbba")
l.append("bbaabbbaa")
l.append("aabbbabba")
l.append("baa-bbbaa")
l.append("-aaabbbaa")
l.append("ba-d--aba")
l.append("a-babbbaa")
l.append("abbbcbaaa")
l.append("aab-aaaba")
l.append("bbbbababa")
l.append("ab--bbabb")
l.append("bb-ababaa")
l.append("aa-babbaa")
l.append("ababa-bba")
l.append("bbaaabbaa")
l.append("aabcbbaba")
l.append("baaa-baaa")
l.append("babbba-aa")
l.append("abaa---aa")
l.append("ab--bbbaa")
l.append("aa-bbbbaa")
l.append("ba-a-aaaa")
l.append("bbbbbaaab")
l.append("ab--bbbbb")
l.append("bb-bbbbaa")
l.append("aa-bbbbaa")
l.append("aba-bbbba")
l.append("bbaabbbaa")
l.append("aabbbabba")
l.append("baa-bbbaa")
l.append("aba-bbbaa")
l.append("ba-b--aba")
l.append("a-babbbaa")
l.append("abbbbaaab")
l.append("aab-aadba")
l.append("bbbbaaabb")
l.append("ab--bbabb")
l.append("bb-ababaa")
l.append("aa-babbaa")
l.append("abaab-bba")
l.append("bbaabbaaa")
l.append("baabqbabb")
l.append("aaaa-abba")
l.append("fdfaaba-b")
l.append("fab-a--aa")
l.append("adfab--bb")
l.append("aacadfxeb")
l.append("ba-dfa-aa")
l.append("bbabbaaab")
l.append("ab-bebgbb")
l.append("bb-babadf")
l.append("aab-bbasd")
l.append("aba-bbbba")
l.append("basdfbaab")
l.append("aasdfabbb")
l.append("badsfaa-b")
l.append("dfa-aaabb")
l.append("badsfa-d-")
l.append("a-adsfbab")
l.append("dafabbbcb")
l.append("fdafaab-a")
l.append("bbadfbbab")
l.append("ab--adfbb")
l.append("bb-aasdfb")
l.append("aa-asdfba")
l.append("ababa-baa")
l.append("bbaaaafab")
l.append("aabcbbaaa")
l.append("baxa-baaa")
l.append("babbax-aa")



#######################################################
#
#   create_y_large_a
#

def create_y_large_a(l, nThreshold):

    y = []

    for line in l:
        count = line.count('a')

        if count >= nThreshold:
            y.append(1.0)
        else:
            y.append(0.0)

    y = np.array(y)
    y = y.astype(np.float32)
    return y

"""c"""


#######################################################
#
#   create_y_a_first
#

def create_y_a_first(l):

    y = []

    for line in l:
        if line[0] == 'a':
            y.append(1.0)
        else:
            y.append(0.0)

    y = np.array(y)
    y = y.astype(np.float32)
    return y

"""c"""


#######################################################
#
#   create_y_contains_d
#

def create_y_contains_d(l):

    y = []

    for line in l:
        count = line.count('d')
        
        if count == 0:
            y.append(0.0)
        else:
            y.append(1.0)

    y = np.array(y)
    y = y.astype(np.float32)
    return y

"""c"""


L = len (l[0])

# Target:

y = create_y_contains_d(l)


n = tokenizer.texts_to_sequences(l)

X = np.array(n, dtype= np.float32)

np.random.seed(137)

splits = 3

kf = KFold(n_splits = splits)
    
nSplits = kf.get_n_splits(X)

nFold = 0

cm_l = np.zeros((2,2), dtype = np.float32)



for train_index, valid_index in kf.split(X):

    print ("FOLD# " + str(nFold))

    X_train = X[train_index]  
    y_train = y[train_index]

    X_test = X[valid_index]
    y_test = y[valid_index]


    model = Sequential()
    model.add(Embedding(6, 6, input_length= L ))

    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(1201))


    #model.add(Dense(L, input_dim=L, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=18)

    y_p = model.predict(X_test)

    y_p = (y_p > 0.5)

    y_p = y_p.astype(float)
    
    cm = confusion_matrix(y_test, y_p)

    cm_l = cm_l + cm

    nFold = nFold + 1

"""c"""

