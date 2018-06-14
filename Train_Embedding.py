

#
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# https://jovianlin.io/keras-models-sequential-vs-functional/
#

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.core import Dense

from keras.layers import concatenate

import numpy as np
import gc

#
#
# Todo: Take char input
# Todo: To regression output. See Driver_SM.py
#


#------------------------------------------------------------------- 

# Classification. Word input. Fully connected network.



labels = np.array([1,1,1,1,1,0,0,0,0,0])

docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!',
                'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.']


# 
# Embeddings for categorical data
#
# http://flovv.github.io/Embeddings_with_keras/
#
#

cats = [3, 2, 0, 2, 2, 0, 1, 1, 1, 7]

num_cats = 1 + np.array(cats).max() - np.array(cats).min()

embedding_size_cat = int (0.5 + np.min([50, num_cats/ 2]))

acCat = np.array(cats)
acCat = acCat.reshape(10, 1)

print (acCat)

num_words = 10

vocab_size = 50

encoded_docs = [one_hot(d, vocab_size) for d in docs]
  
padded_docs = pad_sequences(encoded_docs, maxlen= num_words, padding='post')

print(padded_docs)

input_layer_0 = Input(shape=(num_words,), name = "input_0")
embedding_layer_0 = Embedding(vocab_size, 3, name = "Emb_0")(input_layer_0)

input_layer_1 = Input(shape=(num_words,), name = "input_1")
embedding_layer_1 = Embedding(vocab_size, 2, name = "Emb_1")(input_layer_1)

input_layer_2 = Input(shape=(num_words,), name = "input_2")
embedding_layer_2 = Embedding(vocab_size, 2, name = "Emb_2")(input_layer_2)

input_layer_3 = Input(shape=(num_words,), name = "input_3")
embedding_layer_3 = Embedding(vocab_size, 2, name = "Emb_3")(input_layer_3)

c_layer = concatenate([embedding_layer_0, embedding_layer_1, embedding_layer_2, embedding_layer_3])

flatten_0 = Flatten() (c_layer)

input_layer_cat = Input(shape=(1,), name = "input_cat")
embedding_layer_cat = Embedding(num_cats, embedding_size_cat, name = "Emb_cat")(input_layer_cat)

flatten_cat = Flatten() (embedding_layer_cat)

d_layer = concatenate([flatten_0, flatten_cat])

deep_0 = Dense(10) (d_layer)

deep_1 = Dense(3) (deep_0)

output_layer = Dense(1, activation='sigmoid') (deep_1)

lcInput = [input_layer_0, input_layer_1, input_layer_2, input_layer_3, input_layer_cat]

m = Model(inputs= lcInput, outputs=output_layer)

print (m.summary())

m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

lcInputData = [padded_docs, padded_docs, padded_docs, padded_docs, acCat]

m.fit(lcInputData, [labels], epochs=4, verbose=0)

loss, accuracy = m.evaluate(lcInputData, labels, verbose=0)

print(f"Accuracy: {accuracy*100:.1f}")


# https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/


# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest

top_words = 50000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


max_words = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()

#
# We will use an Embedding layer as the input layer, setting the vocabulary to 5,000,
# the word vector size to 32 dimensions and the input_length to 500.
# 
# 
# The output of this layer will be a 32×500 sized matrix.

model.add(Embedding(top_words, 32, input_length=max_words))

# We will flatten the Embedded layers output to one dimension

model.add(Flatten())

# then use one dense hidden layer of 250 units with a rectifier activation function

model.add(Dense(350, activation='relu'))

# The output layer has one neuron and will use a sigmoid activation to output values of 0 and 1 as predictions.

model.add(Dense(1, activation='sigmoid'))

# The model uses logarithmic loss and is optimized using the efficient ADAM optimization procedure.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=6, batch_size=128, verbose=2)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))





model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

