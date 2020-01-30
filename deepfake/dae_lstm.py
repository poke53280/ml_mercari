

from numpy import array
from keras.models import Sequential
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



def load_part(iPart):
    output_dir = get_output_dir()
    assert output_dir.is_dir()
    l_npy = []
    l_orig = []

    for x in output_dir.iterdir():
        prefix = f"lines_p_{iPart}_"
        if x.suffix == '.npy' and x.stem.startswith(prefix):

            original_name = x.stem.split(prefix, 1)[1]

            # TODO, add original name to every sampled line
            assert len (original_name) > 5
            l_npy.append(np.load(x))
            l_orig.append(original_name)

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


class MyCustomCallback(Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))


anTrain10, l_orig = load_part(10)
anTrain23 = load_part(23)
anTrain0 = load_part(0)


anTrain = np.concatenate([anTrain10, anTrain23, anTrain0])

np.random.shuffle(anTrain)

anTest = load_part(24)

anTrain = preprocess_input(anTrain)
anTest = preprocess_input(anTest)

# Real part only

sequence = anTrain[:, :16, :]

test_sequence_real  = anTest[:, :16, :]
test_sequence_fake  = anTest[:, 16:, :]


num_samples = sequence.shape[0]
num_timesteps = sequence.shape[1]


model = Sequential()
#model.add(LSTM(2048, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
#model.add(LSTM(256, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
model.add(LSTM(8, activation='relu'))

model.add(RepeatVector(num_timesteps))

model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=True))
#model.add(LSTM(256, activation='relu', return_sequences=True))
#model.add(LSTM(2048, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(3)))
model.compile(optimizer='adam', loss='mse')

model.summary()


# fit model
model.fit(sequence[:100000], sequence[:100000], epochs=1, batch_size=2048, verbose=1, callbacks = [MyCustomCallback()])


# Test: On each sampled original (real and fake sampling data)
#           Predict on a set of real lines.
#           Predict on a set of fake lines.



y_test_real = model.predict(test_sequence_real[:1000]).reshape(-1)
mse_real = mean_squared_error(y_test_real[:1000], test_sequence_real.reshape(-1)[:1000])

y_test_fake = model.predict(test_sequence_fake[:1000]).reshape(-1)
mse_fake = mean_squared_error(y_test_fake[:1000], test_sequence_fake.reshape(-1)[:1000])


mse_real
mse_fake

test_sequence_real[:1000]
test_sequence_fake[:1000]



model.save(p / 'my_model.h5')
# del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model(p / 'my_model.h5')

# create random sequence as baseline
y_random = np.random.uniform(size = test_sequence_real.shape)
y_random = y_random.reshape((-1, 16, 3))


reconstruction_error(model, sequence[:20000])
reconstruction_error(model, y_random[:20000])
reconstruction_error(model, test_sequence_real)
reconstruction_error(model, test_sequence_fake)








