
import numpy as np
import pandas as pd


DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 500)



training = pd.read_csv(DATA_DIR + 'training_set.csv')

meta_training = pd.read_csv(DATA_DIR + 'training_set_metadata.csv')

meta_training.sample(5)


len (np.unique(meta_training.object_id))


m = training.object_id == 130779836

training[m]


m = meta_training.object_id == 130779836

meta_training[m]


m = training.object_id == 130779836

q = training[m]

afmjd = np.array(q.mjd)

afmjd.sort()

np.diff(afmjd)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


