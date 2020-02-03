

from numpy import array
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import mean_squared_error

from mp4_frames import get_output_dir
import datetime


####################################################################################
#
#   load_part
#

def load_part(iPart):
    output_dir = get_output_dir()
    assert output_dir.is_dir()
    l_npy = []
    l_orig = []

    for x in output_dir.iterdir():
        prefix = f"lines_p_{iPart}_"
        if x.suffix == '.npy' and x.stem.startswith(prefix):
            data_original = np.load(x)
            original_name = x.stem.split(prefix, 1)[1]
            assert len (original_name) > 5

            l_name = [original_name] * data_original.shape[0]
            
            l_npy.append(data_original)
            l_orig.extend(l_name)

    anData = np.concatenate(l_npy)
    return anData, l_orig


####################################################################################
#
#   preprocess_input
#

def preprocess_input(data):
    data = (data.astype(np.float32) - 256/2.0) / (256/2.0)
    return data


####################################################################################
#
#   reconstruction_error
#

def reconstruction_error(model, data):
    data_p = model.predict(data)
    rms = mean_squared_error(data_p.reshape(-1), data.reshape(-1))
    return rms


####################################################################################
#
#   MyCustomCallback
#

class MyCustomCallback(Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))



def train():

    anTrain10, _ = load_part(10)    
    anTrain23, _ = load_part(23)
    anTrain0, _  = load_part(0)


    anTrain = np.concatenate([anTrain10, anTrain23, anTrain0])

    np.random.shuffle(anTrain)

    anTrain = preprocess_input(anTrain)
  

    # Real part only

    sequence_real = anTrain[:, :16, :]
    sequence_fake = anTrain[:, 16:, :]
    


    num_samples = sequence_real.shape[0]
    num_timesteps = sequence_real.shape[1]


    model = Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(4, activation='relu'))

    model.add(RepeatVector(num_timesteps))

    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))

    model.add(TimeDistributed(Dense(3)))
    model.compile(optimizer='adam', loss='mse')

    model.summary()


    p = get_output_dir()

    # fit models
    # 0.0165
    model.fit(sequence_real[:2000000], sequence_real[:2000000], epochs=1, batch_size=256, verbose=1)
    model.save(p / 'my_model_rr.h5')

    model.fit(sequence_real[:2000000], sequence_fake[:2000000], epochs=1, batch_size=256, verbose=1)
    model.save(p / 'my_model_rf.h5')


    model = Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(4, activation='relu'))

    model.add(RepeatVector(num_timesteps))

    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))

    model.add(TimeDistributed(Dense(3)))
    model.compile(optimizer='adam', loss='mse')

    model.summary()


    model.fit(sequence_fake[:2000000], sequence_real[:2000000], epochs=1, batch_size=256, verbose=1)
    model.save(p / 'my_model_fr.h5')

    # 0.0124
    model.fit(sequence_fake[:2000000], sequence_fake[:2000000], epochs=1, batch_size=256, verbose=1)
    model.save(p / 'my_model_ff.h5')













