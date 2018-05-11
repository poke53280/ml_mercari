
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

import numpy as np

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Flatten


docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']

# define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])


vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

#
# The Embedding has a vocabulary of 50 and an input length of 4.
# 
# We will choose a small embedding space of 8 dimensions.
#

model = Sequential()

model.add(Embedding(vocab_size, 8, input_length=max_length))

# The output from the Embedding layer will be 4 vectors of 8 dimensions each.

# We flatten this to a one 32-element vector to pass on to the Dense output layer.

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

model.fit(padded_docs, labels, epochs=150, verbose=0)

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

print('Accuracy: %f' % (accuracy*100))



