
import numpy as np


# plasticc_seq provides:

anData.shape
anDataConst.shape


sentenceLength = anData.shape[1]

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense

encoder_inputs = Input(shape=(sentenceLength,), name="Encoder_input")

vocab_size = 1 + np.max(anDataConst)

emb = Embedding(output_dim=32, input_dim=vocab_size, name="Embedding") (encoder_inputs)

d0 = Dense(64) (emb)

https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/

