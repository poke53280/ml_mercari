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

labels = np.array([1,1,1,1,1,0,0,0,0,0])

docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!', 'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.']

num_words = 4

vocab_size = 50

encoded_docs = [one_hot(d, vocab_size) for d in docs]
  
padded_docs = pad_sequences(encoded_docs, maxlen= num_words, padding='post')

print(padded_docs)

input_layer_0 = Input(shape=(num_words,), name = "input_0")
embedding_layer_0 = Embedding(vocab_size, 8, name = "Emb_0")(input_layer_0)


input_layer_1 = Input(shape=(num_words,), name = "input_1")
embedding_layer_1 = Embedding(vocab_size, 4, name = "Emb_1")(input_layer_1)

c_layer = concatenate([embedding_layer_0, embedding_layer_1])

flatten_0 = Flatten() (c_layer)

output_layer = Dense(1, activation='sigmoid') (flatten_0)

m = Model(inputs=[input_layer_0, input_layer_1], outputs=output_layer)

print (m.summary())


m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

m.fit([padded_docs, padded_docs], [labels], epochs=150, verbose=0)

loss, accuracy = m.evaluate([padded_docs, padded_docs], labels, verbose=0)

print(f"Accuracy: {accuracy*100:.2f}")


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

model.add(Embedding(top_words, 32f, input_length=max_words))

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

