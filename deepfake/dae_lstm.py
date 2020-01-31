

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



anTrain10, _ = load_part(10)
anTrain23, _ = load_part(23)
anTrain0, _  = load_part(0)


anTrain = np.concatenate([anTrain10, anTrain23, anTrain0])

np.random.shuffle(anTrain)

anTest, l_test_orig = load_part(24)

assert anTest.shape[0] == len (l_test_orig)

anTrain = preprocess_input(anTrain)
anTest = preprocess_input(anTest)

# Real part only

sequence = anTrain[:, :16, :]

test_sequence_real  = anTest[:, :16, :]
test_sequence_fake  = anTest[:, 16:, :]


num_samples = sequence.shape[0]
num_timesteps = sequence.shape[1]


model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu'))

model.add(RepeatVector(num_timesteps))

model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(3)))
model.compile(optimizer='adam', loss='mse')

model.summary()


# fit model
model.fit(sequence[:1000000], sequence[:1000000], epochs=1, batch_size=256, verbose=1)


azOrig = np.array(l_test_orig)

azOrigUnique = np.unique(azOrig)
aiOrig = np.searchsorted(azOrigUnique, azOrig)

for iOrig in range (np.max(aiOrig)):

    m_0 = aiOrig == iOrig

    iOrig_real = test_sequence_real[m_0]
    iOrig_fake = test_sequence_fake[m_0]

    y_test_real = model.predict(iOrig_real).reshape(-1)
    mse_real = mean_squared_error(y_test_real, iOrig_real.reshape(-1))

    y_test_fake = model.predict(iOrig_fake).reshape(-1)
    mse_fake = mean_squared_error(y_test_fake, iOrig_fake.reshape(-1))

    print(f"{iOrig}: real {mse_real:.3f} fake {mse_fake:.3f}")







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








