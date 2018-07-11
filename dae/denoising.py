
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
from keras.optimizers import SGD
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.layers import Conv1D
from keras.layers import MaxPool1D, Flatten
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


import time
import gc


########################################################################
#
#    trainCV
#
#

def trainCV(model, y, num_epochs, noise_factor, num_folds, n_batchsize, num_sleep_secs, model_file):

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)

    lKF = list (enumerate(kf.split(y)))

    lRMS = []

    while len(lKF) > 0:
        iLoop, (train_index, test_index) = lKF.pop(0)

        print(f"--- Fold: {iLoop + 1}/ {num_folds} ---")

        # Clear model to random/init
        model.load_weights(model_file)
   

        for i in range(num_epochs):
            print(f"Epoch {i + 1} / {num_epochs}...")

            y_train = y [train_index]
            y_valid = y[test_index]    

            X_train = y[train_index]
            X_valid = y[test_index]

            X_train = add_swap_noise(X_train, y, noise_factor)
            X_valid = add_swap_noise(X_valid, y, noise_factor)

            p_t = np.random.permutation(len(y_train))
            p_v = np.random.permutation(len(y_valid))

            X_train = X_train[p_t]
            y_train = y_train[p_t]

            X_valid = X_valid[p_v]
            y_valid = y_valid[p_v]

            if i > 0:
                # GPU cooling
                print(f"Cooling down for {num_sleep_secs}s...")
                time.sleep(num_sleep_secs)

            h = model.fit(x=X_train, y=y_train, batch_size=n_batchsize, epochs=1, verbose=1, validation_data = (X_valid, y_valid))
            y_p = model.predict(X_valid)

            mse_error = mean_squared_error(y_p, y_valid)

            print(f"Epoch finished. MSE = {mse_error}.")
    
        lRMS.append(mse_error)

    anRMS = np.array(lRMS)

    return anRMS
    
"""c"""

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

def loadY():

    train = pd.read_csv(DATA_DIR + 'train.csv')
    train = train.drop(['target', 'ID'], axis = 1)

    test = pd.read_csv(DATA_DIR + 'test.csv')
    test = test.drop(['ID'], axis = 1)

    df = pd.concat([train, test], axis = 0)

    df = df.apply(dae.RankGauss.gauss_rank_transform, axis = 0, raw = True)

    Y = np.array(df)

    return Y


Y = loadY()

gc.collect()

num_features = Y.shape[1]

input_user = Input(shape=(num_features,))

encoded_l_0 = Dense(1500, activation='relu') (input_user)
encoded_l_1 = Dense(1500, activation='relu') (encoded_l_0)
encoded_l_2 = Dense(1500, activation='relu') (encoded_l_1)

decoded = Dense(num_features, activation='linear') (encoded_l_2)

autoencoder = Model(input_user, decoded)

autoencoder.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.003, decay = 0.995))

autoencoder.summary()

model_file_name = DATA_DIR + 'init_model3.h5'

autoencoder.save_weights(model_file_name)        

anMSE = trainCV(autoencoder, Y, 80, 0.11, 7, 128, 15.0, model_file_name)

MSEmean = anMSE.mean()
MSEstd  = anMSE.std()
    
print(f"  ==> MSE = {MSEmean} +/- {MSEstd}")

#
# swap 0.12 epoch 4 -   0.0615 - val_loss: 0.0622
#
# swap 0.11 epoch 4 -   0.0604 - val_loss: 0.0611
#           epoch 7 -   0.0584 - val_loss: 0.0593
#          epoch 20 -   0.0564 - val_loss: 0.0571
#          epoch 26 -   0.0560 - val_loss: 0.0568
#          epoch 35 -   0.0557 - val_loss: 0.0565
#          epoch 40 -   0.0556 - val_loss: 0.0564
#          epoch 50 -   0.0554 - val_loss: 0.0562
#          epoch 69 -   0.0553 - val_loss: 0.0560
#          epoch 73 -   0.0553 - val_loss: 0.0559
#          epoch 77 -   0.0552 - val_loss: 0.0559
#          epoch 78 -   0.0552 - val_loss: 0.0558
# epoch 79
#
#
# epoch 18  - loss: 0.2027 - val_loss: 0.1975
# epoch 36  - loss: 0.0852 - val_loss: 0.0844
# epoch 45  - loss: 0.0718 - val_loss: 0.0722
# epoch 48  - loss: 0.0704 - val_loss: 0.0709
# epoch 53  - loss: 0.0683 - val_loss: 0.0689
#
# epoch 58 - loss: 0.0671 - val_loss: 0.0678
# epoch 62 - loss: 0.0670 - val_loss: 0.0676
# epoch 67 - loss: 0.0666 - val_loss: 0.0675
# epoch 76 - loss: 0.0664 - val_loss: 0.0673
# epoch 79 - loss: 0.0663 - val_loss: 0.0671
#


>#46115/46115 [==============================] - 29s 627us/step - loss: 0.0663 - val_loss: 0.0672
Epoch finished. MSE = 0.06716904180225143.

--- Fold: 3/ 7 ---


 0.5263 - val_loss: 0.5102

 loss: 0.5117 - val_loss: 0.4999

 6115/46115 [==============================] - 29s 625us/step - loss: 0.5026 - val_loss: 0.4920
 5: 46115/46115 [==============================] - 29s 629us/step - loss: 0.4950 - val_loss: 0.4847

 6: 46115/46115 [==============================] - 29s 628us/step - loss: 0.4806 - val_loss: 0.4705
 7: s 629us/step - loss: 0.4731 - val_loss: 0.4630

 9; oss: 0.4572 - val_loss: 0.4466

 15: 29s 626us/step - loss: 0.4036 - val_loss: 0.3934
 20: 3574 - val_loss: 0.3479
      
 29: 46115/46115 [==============================] - 29s 628us/step - loss: 0.2975 - val_loss: 0.2892

30: loss: 0.2746 - val_loss: 0.2667

38 - loss: 0.2215 - val_loss: 0.2152

42- loss: 0.1993 - val_loss: 0.1935

45 - loss: 0.1841 - val_loss: 0.1787

52  loss: 0.1541 - val_loss: 0.1498

59: loss: 0.1366 - val_loss: 0.1325

61: loss: 0.1247 - val_loss: 0.1209

63: loss: 0.1193 - val_loss: 0.1160

67: loss: 0.1096 - val_loss: 0.1064

69: loss: 0.1055 - val_loss: 0.1024

75: loss: 0.0946 - val_loss: 0.0922

78: loss: 0.0902 - val_loss: 0.0873

79: loss: 0.0889 - val_loss: 0.0863
 29s 623us/step - loss: 0.0875 - val_loss: 0.0851
 -----------------------------------------------------------------------------
NEW FOLD 4

ss: 0.5311 - val_loss: 0.5196

2: loss: 0.5203 - val_loss: 0.5121

3: loss: 0.5137 - val_loss: 0.5061
4  loss: 0.5083 - val_loss: 0.5012

5 loss: 0.5036 - val_loss: 0.4967


 8;  26us/step - loss: 0.4906 - val_loss: 0.4844
13:             loss: 0.4684 - val_loss: 0.4621


20:  loss: 0.4318 - val_loss: 0.4255


24: 27s 593us/step - loss: 0.4091 - val_loss: 0.4030

 28; - loss: 0.3861 - val_loss: 0.3803

33: 46115/46115 [==============================] - 27s 591us/step - loss: 0.3575 - val_loss: 0.3518
Epoch finished. MSE = 0.3518483158051623.


 37: loss: 0.3353 - val_loss: 0.3302
# swa0 0.15

39: loss: 0.3245 - val_loss: 0.3198

43: 115/46115 [==============================] - 28s 598us/step - loss: 0.3037 - val_loss: 0.2991


57: loss: 0.2400 - val_loss: 0.2367
60: loss: 0.2280 - val_loss: 0.2250
61: loss: 0.2242 - val_loss: 0.2212

66: loss: 0.2062 - val_loss: 0.2037

68: loss: 0.1995 - val_loss: 0.1970

# Working: 
# epoch 80: loss: 0.0608 - val_loss: 0.0615
#
#
# Adam. lr = 0.003, decay = 0.995
#input_user = Input(shape=(num_features,))
#
#out = Dense(1500, activation='relu') (input_user)
#out = Dense(1500, activation='relu') (out)
#out = Dense(1500, activation='relu') (out)
#out = Dense(num_features, activation='linear') (out)
# X_train = add_swap_noise(X_train, y, 0.15)
#  h = model.fit(x=X_train, y=y_train, batch_size=128, epochs=1, verbose=1, validation_data = (X_valid, y_valid))
#
#
# OOM on biggest example
#
#
