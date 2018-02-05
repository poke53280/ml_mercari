

# A study of 
# https://www.kaggle.com/vsmolyakov/keras-cnn-with-fasttext-embeddings
# by kaggler vsmolyakov
#
#
# Keras CNN with FastText Embeddings
#
#
# CNNs provide a faster alternative to LSTM models at a comparable performance. They are faster to train and use fewer parameters.
# 
# CNN models are translation invariant and in application to text make sense when there is no strong dependence on recent past
# vs distant past of the input sequence.
# 
# CNNs can learn patterns in word embeddings and given the nature of the dataset (e.g. multple misspellings, out of vocabulary words),
# it makes sense to use sub-word information. In this notebook, a simple CNN architecture is used for multi-label classification
# with the help of FastText word embeddings. Thus, it can be a good addition to your ensemble.
#
#

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs



