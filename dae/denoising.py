
#
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
#

from dae.SwapNoise import add_swap_noise
import dae.RankGauss
import numpy as np
import pandas as pd
import scipy 
from scipy.sparse import vstack
from keras.layers import Input, Dense
from keras.models import Model

from keras.optimizers import Adam
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.layers import Conv1D
from keras.layers import MaxPool1D, Flatten
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'train.csv')
train = train.drop(['target', 'ID'], axis = 1)

test = pd.read_csv(DATA_DIR + 'test.csv')
test = test.drop(['ID'], axis = 1)

df = pd.concat([train, test], axis = 0)

q = df.apply(dae.RankGauss.gauss_rank_transform, axis = 0, raw = True)

q.describe()

y = np.array(q)


num_features = y.shape[1]

input_user = Input(shape=(num_features,))

out = Dense(32, activation='relu') (input_user)
out = Dense(num_features, activation='linear') (out)

model = Model(input_user, out)

model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.01))

model.summary()

model.save_weights(DATA_DIR + 'init_model.h5')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

NUM_FOLDS = 7

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

lKF = list (enumerate(kf.split(y)))

lRMS = []

while len(lKF) > 0:
    iLoop, (train_index, test_index) = lKF.pop(0)

    print(f"--- Fold: {iLoop + 1}/ {NUM_FOLDS} ---")

    # Clear model to random/init
    model.load_weights(DATA_DIR + 'init_model.h5')

    y_train = y [train_index]
    y_valid = y[test_index]    

    for i in range(250):
        print(f"Epoch {i + 1}...")

        X_train = y[train_index]
        X_valid = y[test_index]

        X_train = add_swap_noise(X_train, y, 0.0)
        X_valid = add_swap_noise(X_valid, y, 0.0)

        np.random.shuffle(X_train)
        np.random.shuffle(X_valid)

        h = model.fit(x=X_train, y=y_train, batch_size=32, epochs=1, verbose=1, validation_data = (X_valid, y_valid))
        y_p = model.predict(X_valid)
        mse_error = mean_squared_error(y_p, y_valid)
        print (mse_error)
    
    lRMS.append(mse_error)
    
"""c"""
anRMS = np.array(lRMS)

RMSLEmean = anRMS.mean()
RMSLEstd  = anRMS.std()

print(f"  ==> RMSLE = {RMSLEmean} +/- {RMSLEstd}")


# One pre-generated set. ==> RMSLE = 0.43662030341314306 +/- 0.153089487053149

# New noise per epoch. 250 epochs, 7 fold 1500,1500,1500

