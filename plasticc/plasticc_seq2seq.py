
import numpy as np
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense

from keras.models import Model

import tensorflow as tf

# plasticc_seq provides:
#anData.shape
#anDataConst.shape


num_rows = 7
sentenceLength = 6

vocab_size = 5

anDataConst = np.random.randint(0, vocab_size, size = (num_rows, sentenceLength), dtype = np.uint16)
anData = anDataConst.copy()


emb_obj = Embedding(output_dim=3, input_dim=vocab_size, name="Embedding")

encoder_inputs = Input(shape=(sentenceLength,), name="Encoder_input")
target_inputs = Input(shape=(sentenceLength,), name="target_input")

x = emb_obj (encoder_inputs)

x = Dense(64) (x)

x = Dense(64) (3 * vocab_size)

# Input string is now from embedding to same shape as embedding

# Target string is now from embedding
t = emb_obj(target_inputs)

model = Model(x, t)

diff = 'x - t' ex: rmse


#
# Train so that x and t are close/zero.
#
# When close:
#
# Network can deliver/predict the embedding of target value / after reverse map deliver the target value.
#
#


# https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/

# Scalable Sequence-to-Sequence Problem



from random import randint
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]



# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# generate a single source and target sequence
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
print(X1.shape, X2.shape, y.shape)
print('X1=%s, X2=%s, y=%s' % (one_hot_decode(X1[0]), one_hot_decode(X2[0]), one_hot_decode(y[0])))



X1[0]


from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
        X1, X2, y = list(), list(), list()
        for _ in range(n_samples):
                # generate source sequence
                source = generate_sequence(n_in, cardinality)
                # define padded target sequence
                target = source[:n_out]
                target.reverse()
                # create padded input target sequence
                target_in = [0] + target[:-1]
                # encode
                src_encoded = to_categorical([source], num_classes=cardinality)
                tar_encoded = to_categorical([target], num_classes=cardinality)
                tar2_encoded = to_categorical([target_in], num_classes=cardinality)
                # store
                X1.append(src_encoded)
                X2.append(tar2_encoded)
                y.append(tar_encoded)
        X1 = np.squeeze(array(X1), axis=1) 
        X2 = np.squeeze(array(X2), axis=1) 
        y = np.squeeze(array(y), axis=1) 
        return X1, X2, y

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# define model
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
# train model
train.fit([X1, X2], y, epochs=1)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1

print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

# spot check some examples
for _ in range(10):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))

















# https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do


vocab = ['the','like','between','did','just','national','day','country','under','such','second']


emb = np.array([[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.044457, -0.49688, -0.17862],
   [0.36808, 0.20834, -0.22319, 0.046283, 0.20098, 0.27515, -0.77127, -0.76804],
   [0.7503, 0.71623, -0.27033, 0.20059, -0.17008, 0.68568, -0.061672, -0.054638],
   [0.042523, -0.21172, 0.044739, -0.19248, 0.26224, 0.0043991, -0.88195, 0.55184],
   [0.17698, 0.065221, 0.28548, -0.4243, 0.7499, -0.14892, -0.66786, 0.11788],
   [-1.1105, 0.94945, -0.17078, 0.93037, -0.2477, -0.70633, -0.8649, -0.56118],
   [0.11626, 0.53897, -0.39514, -0.26027, 0.57706, -0.79198, -0.88374, 0.30119],
   [-0.13531, 0.15485, -0.07309, 0.034013, -0.054457, -0.20541, -0.60086, -0.22407],
   [ 0.13721, -0.295, -0.05916, -0.59235, 0.02301, 0.21884, -0.34254, -0.70213],
   [ 0.61012, 0.33512, -0.53499, 0.36139, -0.39866, 0.70627, -0.18699, -0.77246 ],
   [ -0.29809, 0.28069, 0.087102, 0.54455, 0.70003, 0.44778, -0.72565, 0.62309 ]])


emb.shape

from collections import OrderedDict


# embedding as TF tensor (for now constant; could be tf.Variable() during training)
tf_embedding = tf.constant(emb, dtype=tf.float32)

# input for which we need the embedding

input_str = "like the country"


word_to_idx = OrderedDict({w:vocab.index(w) for w in input_str.split() if w in vocab})


embed = tf.nn.embedding_lookup(tf_embedding, list(word_to_idx.values()))

session = tf.Session()

result = session.run(embed)

print(result)



