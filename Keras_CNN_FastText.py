

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


# So far I've had best luck with simple FastText and GRUs for text fields and separate embeddings for
# categorical inputs. With these I get about 15 minutes for 0.42 and 7 minutes for 0.425 scores.
# Convolutions don't do much, neither do MaxPooling layers.








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


sns.set_style("whitegrid")
np.random.seed(0)

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\toxic\\"

MAX_NB_WORDS = 10000


print('loading word embeddings...')

embeddings_index = {}

f = codecs.open(DATA_DIR_PORTABLE + 'wiki.simple.vec', encoding='utf-8')

for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

"""c"""

print('found %s word vectors' % len(embeddings_index))

#load data
train_df = pd.read_csv(DATA_DIR_PORTABLE + 'train.csv', sep=',', header=0)
test_df = pd.read_csv(DATA_DIR_PORTABLE + 'test.csv', sep=',', header=0)
test_df = test_df.fillna('_NA_')

print("num train: ", train_df.shape[0])
print("num test: ", test_df.shape[0])

label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train_df[label_names].values

train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))

max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)


sns.distplot(train_df['doc_len'], hist=True, kde=True, color='b', label='doc len')

plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')

plt.title('comment length'); plt.legend()

plt.show()

raw_docs_train = train_df['comment_text'].tolist()
raw_docs_test = test_df['comment_text'].tolist() 

num_classes = len(label_names)

print("pre-processing train data...")



def preprocess(raw_docs):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    processed_docs = []

    for doc in tqdm(raw_docs):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs.append(" ".join(filtered))

    return processed_docs

"""c"""

processed_docs_train = preprocess(raw_docs_train)
processed_docs_test = preprocess(raw_docs_test)


"""c"""
#end for

print("tokenizing input data...")


tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)

tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky

word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))



sample = 2321
print (word_seq_test[sample])
len (word_seq_test[sample])

print (processed_docs_test[sample])
len (processed_docs_test[sample].split(" "))

# pad sequences:


word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

#training params
batch_size = 256 
num_epochs = 8 

#model parameters
num_filters = 64 
embed_dim = 300 
weight_decay = 1e-4

# We can now prepare our embedding matrix limiting to a max number of words

print('preparing embedding matrix...')

words_not_found = []

nb_words = min(MAX_NB_WORDS, len(word_index))


embedding_matrix = np.zeros((nb_words, embed_dim))



for word, i in word_index.items():
    
    if i >= nb_words:
        # Not among the most frequent words in the corpus.
        continue

    # Find data on this word in the database    
    embedding_vector = embeddings_index.get(word)

    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # The word from the input has got an embedding vector
        embedding_matrix[i] = embedding_vector
    else:
        # embedding_matrix[i] will remain at all zeros as initialized.
        words_not_found.append(word)

"""c"""


print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#CNN architecture
print("training CNN ...")

model = Sequential()

model.add(Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False))


model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))


model.add(MaxPooling1D(2))


model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))



model.add(GlobalMaxPooling1D())


model.add(Dropout(0.5))


model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))



model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)

callbacks_list = [early_stopping]


hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)


y_test = model.predict(word_seq_test)


###########################################################################


df['item_description'].fillna(value='missing', inplace=True)

df['doc_len'] = df.item_description.apply(lambda words: len(words.split(" ")))

max_seq_len = np.round(df['doc_len'].mean() + df['doc_len'].std()).astype(int)

processed_docs_mercari = preprocess(df.item_description)

tokenizer.fit_on_texts(processed_docs_mercari)

word_seq_mercari = tokenizer.texts_to_sequences(processed_docs_mercari)

