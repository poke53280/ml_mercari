
DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

# Assume AUC 0.87

from sklearn.metrics import log_loss

import numpy as np

y_true = np.array([[0, 1, 0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0]])

y_pred = np.array([[0.0, 0.8, 0.1, 0.1,0,0,0,0,0,0], [0.1, 0.1, 0.8, 0.0,0,0,0,0,0,0] ])

log_loss(y_true, y_pred)

meta = pd.read_csv(DATA_DIR + 'test_set_metadata.csv')

print(f"Number of test items: {meta.shape[0]}")

data = pd.read_pickle(DATA_DIR + 'test_processedtest920_0.pkl')

data.shape

#
#Dataset generation done.
#Time spent generating 10 x 17465: 396.1s.  rows/s: 440.9 items/s: 44.1
#   
#Saved dataset as 'C:\plasticc_data\test_processedtest920_0.pkl'
#
# Many lines generation very fast when at the item.
#
# => Prediction for all 3492890 items: 22.0h  with 10x oversampling singhe threded
#
#* Very slow: ----4 => 5 - and takes memory.
#
#* Print 'saving'
#
# Status at 11.05: => All in about 8 hours on this computer given the 4 => 5 resolved.
#


