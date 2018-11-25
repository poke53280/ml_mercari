
import numpy as np
import pandas as pd
import gc
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Flatten

from keras.layers import Subtract
from keras.layers import LSTM
from keras.layers import TimeDistributed


from keras.models import Model

from keras.constraints import unitnorm

import tensorflow as tf
from sklearn.model_selection import KFold



def add_noise(anDataConst, anData, value_area, anSnippet, aSnippetSize):

    num_objects = anDataConst.shape[0]

    sample_init_size = anSnippet.shape[1]


    for iRow in range(num_objects):

        iMin = 0
        iMax = value_area[iRow] - sample_init_size

        iOffset = np.random.choice(range(iMin, iMax + 1))

        assert iOffset >= 0
        assert iOffset + sample_init_size <= anDataConst.shape[1]

        data_raw = anDataConst[iRow, iOffset: iOffset + sample_init_size]

        iFirstBreakChar = 6 * num_bins_y

        m = data_raw >= iFirstBreakChar

        iCut = sample_init_size

        if m.sum() > 0:
            iCut = np.where(m)[0][0]

        # Can replace values 0..iCut

        iSnippetRow = np.random.randint(0, anSnippet.shape[0])

        anSnippetData = anSnippet[iSnippetRow]
        nSnippetSize = aSnippetSize[iSnippetRow]

        nReplaceRun = np.amin([nSnippetSize, iCut])

        anData[iRow, iOffset: iOffset + nReplaceRun] = anSnippetData[0:nReplaceRun]


    m = anData == anDataConst

    nEqual = m.sum()
    nAll = anData.shape[0] * anData.shape[1]

    nDiff = nAll - nEqual
    rDiff = 100.0 * nDiff / nAll

    print(f"add_noise diff elements: {rDiff:.1f}%")

"""c"""



######## LOAD START



DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

num_snip_per_row = 10
slot_size = 5
num_bins_y = 150
num_sequence_length = 200

anDataConst = np.load(DATA_DIR + "anData_all.npy")
value_area = np.load(DATA_DIR + "value_area_all.npy")

NCUT = 1000000

anDataConst = anDataConst[:NCUT]
value_area = value_area[:NCUT]

gc.collect()

# Load snippets

anSnippet = np.load(DATA_DIR + "anSnippet_all.npy")
aSnippetSize = np.load(DATA_DIR + "aSnippetSize_all.npy")

anSnippet = anSnippet[:NCUT* num_snip_per_row]
aSnippetSize = aSnippetSize[:NCUT* num_snip_per_row]

gc.collect()

vocab_size = np.max(anDataConst) + 1 

sentenceLength = anDataConst.shape[1]
num_rows = anDataConst.shape[0]

ac = np.linspace(start = 0, stop = 1, num = 150)

nBreaks = 1 + np.max(anDataConst) -  6 * num_bins_y

ab = np.linspace(start = 0, stop = 1, num = nBreaks)

emb = np.zeros((vocab_size, 7), dtype = np.float32)

for b in range(6):
    print(f"{b * 150} - {(b+1) * 150}")
    emb[b * 150: (b+1) * 150, b] = ac

"""c"""

emb[6* 150: 6*150 + nBreaks, 6] = ab

anZeroDiff = np.zeros( (num_rows, sentenceLength * 7), dtype = np.float32)

num_folds = 9
lRunFolds = list (range(num_folds))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)

lKF = list (enumerate(kf.split(anDataConst)))

lMSE = []

iFold = 0


for iFold in range(1):

    print(f"iFold {iFold}")

    iLoop, (train_index, test_index) = lKF[iFold]

    anDataConstTrain = anDataConst[train_index]
    anDataConstValid = anDataConst[test_index]


    isAddNoise = False

    if isAddNoise:

        anSnipIDxTrain = np.empty((train_index.shape[0], num_snip_per_row), dtype = int)
        anSnipIDxTrain[:, 0] = train_index
        anSnipIDxTrain[:, 0] *= num_snip_per_row

        for i in range(1, num_snip_per_row):
            anSnipIDxTrain[:, i] = anSnipIDxTrain[:, i-1] + 1
        """c"""

        anSnipIDxTrain = anSnipIDxTrain.flatten(order = 'C')

        anSnippetTrain = anSnippet[anSnipIDxTrain]
        aSnippetSizeTrain = aSnippetSize[anSnipIDxTrain]
  

        anSnipIDxValid = np.empty((test_index.shape[0], num_snip_per_row), dtype = int)
        anSnipIDxValid[:, 0] = test_index
        anSnipIDxValid[:, 0] *= num_snip_per_row

        for i in range(1, num_snip_per_row):
            anSnipIDxValid[:, i] = anSnipIDxValid[:, i-1] + 1
        """c"""

        anSnipIDxValid = anSnipIDxValid.flatten(order = 'C')

        anSnippetValid = anSnippet[anSnipIDxValid]
        aSnippetSizeValid = aSnippetSize[anSnipIDxValid]

    encoder_inputs = Input(shape=(sentenceLength,), name="Encoder_input")
    target_inputs = Input(shape=(sentenceLength,), name="target_input")

    emb_obj = Embedding(output_dim=emb.shape[1], input_dim=emb.shape[0], name="Embedding", embeddings_constraint=unitnorm(axis=1))

    #emb_obj = Embedding(emb.shape[0], emb.shape[1], weights=[emb], trainable=False)

    x = emb_obj (encoder_inputs)

    x = Flatten() (x)

    x = Dense(1400) (x)
    x = Dense(1400) (x)
    

    # 45 mins. on high noise. (shuffle 8)
    # Valid set MSE = 0.0335
    x = LSTM(128, return_sequences=True) (x)
    x = LSTM(128, return_sequences=True) (x)
    x = TimeDistributed(Dense(vocab_size)) (x)
    x = Flatten() (x)
    x = Dense(1) (x)

    x = LSTM(128, return_sequences=True) (x)
    x = TimeDistributed(Dense(vocab_size)) (x)
    x = Dense(512) (x)
    x = Flatten() (x)
    x = Dense(32) (x)
    x = Dense(1) (x)

    
    x = Flatten() (x)
    x = Dense(8) (x)
    x = Dense(1) (x)

    #emb_obj_t = Embedding(output_dim=num_dim_out_emb, input_dim=vocab_size, name="Embedding_T", embeddings_constraint=unitnorm(axis=1))

    #assert hasattr(emb_obj_t, 'trainable')

    #emb_obj_t.trainable = False

    t = emb_obj (target_inputs)
   
    t = Flatten() (t)

    subtracted = Subtract()([x, t])

    model = Model([encoder_inputs, target_inputs], subtracted)

    model.compile(loss='mse', optimizer='adam')

    num_shuffles = 8

    num_epochs = 20

#     emb_obj_t.set_weights(emb_obj.get_weights())
    

    for iEpoch in range(num_epochs):

        print(f"Epoch {iEpoch + 1}/ {num_epochs}")

        anDataTrain = anDataConstTrain.copy()
        anDataValid = anDataConstValid.copy()

        if isAddNoise:
            for iShuffle in range(num_shuffles):
                add_noise(anDataConstTrain, anDataTrain, value_area[train_index], anSnippetTrain, aSnippetSizeTrain)
                add_noise(anDataConstValid, anDataValid, value_area[test_index], anSnippetValid, aSnippetSizeValid)
        """c"""

        num_train = anDataTrain.shape[0]

        max_commit = 100000

        n_commit_split = 1 + int (num_train / max_commit)

        idxTrain = np.arange(0, anDataTrain.shape[0])
        idxValid = np.arange(0, anDataValid.shape[0])

        idxTrainSplit = np.array_split(idxTrain, n_commit_split, axis = 0)
        idxValidSplit = np.array_split(idxValid, n_commit_split, axis = 0)

        for ix in range(n_commit_split):
            print(f"ix = {ix} of {n_commit_split}")
            num_train = anDataTrain[idxTrainSplit[ix]].shape[0]
            num_valid = anDataValid[idxValidSplit[ix]].shape[0]

            assert anZeroDiff.shape[0] >= num_train
            assert anZeroDiff.shape[0] >= num_valid

            h = model.fit(batch_size=16, x = [anDataTrain[idxTrainSplit[ix]],  anDataConstTrain[idxTrainSplit[ix]]], y = anZeroDiff[:num_train], validation_data = ([anDataValid[idxValidSplit[ix]],  anDataConstValid[idxValidSplit[ix]]], anZeroDiff[:num_valid]), epochs = 1, verbose = 1)
           
        """c"""

        z_p = model.predict(x = [anDataValid[idxValidSplit[8]],  anDataConstValid[idxValidSplit[8]]])

        mse = (z_p* z_p).mean()

        print(f"Valid set MSE = {mse:.4f}")


    """c"""
"""c"""
    


anDataValid
anDataConstValid


# Unit model

encoder_inputs = Input(shape=(sentenceLength,), name="Encoder_input")
target_inputs = Input(shape=(sentenceLength,), name="target_input")

emb_obj = Embedding(emb.shape[0], emb.shape[1], weights=[emb], trainable=False)

x = emb_obj (encoder_inputs)
x = Flatten() (x)

t = emb_obj (target_inputs)
t = Flatten() (t)

subtracted = Subtract()([x, t])

model_unit = Model([encoder_inputs, target_inputs], subtracted)

model_unit.compile(loss='mse', optimizer='adam')


model_unit.summary()

z_p = model_unit.predict(x = [anDataValid,  anDataConstValid])

mse = (z_p* z_p).mean()

print(f"Valid set MSE = {mse:.4f}")


    
    # M = emb_obj.get_weights()[0]
    


afMSE = np.array(lMSE)

print(f"MSE for all folds: {np.mean(afMSE)} +/ {np.std(afMSE)}")

# emb 6 dense 8 dense 1
#
#
# 500 epochs. Fold 0
# loss: 0.1396 - val_loss: 0.1400
# Improving on stop
# + 500:
# loss: 0.1323 - val_loss: 0.1327
# Improvinb a bit
# + 500:
# loss: 0.1232 - val_loss: 0.1237
# + 500:
# loss: 0.1118 - val_loss: 0.1124
# + 500:
# loss: 0.0985 - val_loss: 0.0991
# + 500:
# loss: 0.0837 - val_loss: 0.0844
# + 500:
# loss: 0.0684 - val_loss: 0.0691
# + 500:
# loss: 0.0539 - val_loss: 0.0545
# + 500:
# loss: 0.0410 - val_loss: 0.0415
# + 500:
# loss: 0.0303 - val_loss: 0.0308
# + 500:
# loss: 0.0220 - val_loss: 0.0224
# + 500:
# loss: 0.0159 - val_loss: 0.0162
# + 500:
# loss: 0.0115 - val_loss: 0.0117
# + 500:
# loss: 0.0084 - val_loss: 0.0086
# + 500:
# loss: 0.0062 - val_loss: 0.0063
# + 500:
# loss: 0.0047 - val_loss: 0.0048
# + 500:
# loss: 0.0035 - val_loss: 0.0036
# +5000:
# loss: 4.2427e-04 - val_loss: 4.5462e-04





# 100 epochs emb 6 dense 512 dense 1
# Improving on stop




x_pred = np.array([2, 3, 2, 4, 1, 2], dtype = np.uint16)

x_pred = x_pred.reshape(1, -1)

t_pred = x_pred

t_pred[:] = 3


z_pred = model.predict([x_pred, t_pred])


rms_error = np.abs(z_pred).mean()

rms_error

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



