
#########
#
#       Further preprocessing
#
#

import pandas as pd
import numpy as np


DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


df = pd.read_pickle(DATA_DIR + "noised_30JUL2018.pkl")

# Convert data types

###### MD ######################################################################
#
# No nan in uint16 range

aMD = np.array(df.MD)

aMD = aMD.astype('uint16')

df = df.assign(MD = aMD)

###### D  #########################################################################
# No nan, in uint16 range

aD = np.array(df.D)

aD = aD.astype('uint16')
df = df.assign(D = aD)

############# B ##################################################################

aB = np.array(df.B)

rNullPct = 100.0 * np.isnan(aB).sum() / aB.shape[0]
print(f"Null b: {rNullPct:.1f}%")

np.nanmax(aB)   ### Error? - very recent
18379.0

np.nanmin(aB)
-16673.0

############## S ###############################################################

aS = np.array(df.S)

rNullPct = 100.0 * np.isnan(aS).sum() / aS.shape[0]
print(f"Null s: {rNullPct:.1f}%")

mB = np.isnan(aB)
mS = np.isnan(aS)

m = (mS == mB)

# B and S is NaN for the same rows:
assert ((m == False).sum() == 0)


m_NAN = np.isnan(aS)

m_G0 = (aS == 0)
m_G1 = (aS == 1)


aS[m_NAN] = 0

aS[m_G0] = 100
aS[m_G1] = 101


np.unique(aS)

aS = aS.astype('uint8')

df = df.assign(S = aS)


### F0, F1, T0 ######################################################################

aF0 = np.array(df.F0_noised)
aF1 = np.array(df.F1_noised)
aT0 = np.array(df.T0_noised)

aC = np.concatenate([aF0, aF1, aT0])

assert (np.isnan(aC).sum() == 0)

np.max(aC)

18851.0

np.min (aC)

6028


# All epochs.
# 
# Need Nan slot and
#
# Min = -16673.0
# Max = 18851
#
#
# Convert B, F0, F1, T0:
#
# 0 : NAN

# Value = value + 16673 + 100

# Max => 35624

m = np.isnan(aB)

aB[m] = 0

aB[~m] = aB[~m] + 16673 + 100

aF0 = aF0 + 16673 + 100
aF1 = aF1 + 16673 + 100
aT0 = aT0 + 16673 + 100


aB = aB.astype('uint16')
aF0 = aF0.astype('uint16')
aF1 = aF1.astype('uint16')
aT0 = aT0.astype('uint16')


df = df.assign(B = aB)
df = df.assign(F0_noised = aF0)
df = df.assign(F1_noised = aF1)
df = df.assign(T0_noised = aT0)


aSeq = np.array(df.SeqID)

np.isnan(aSeq).sum()

np.min(aSeq)
np.max(aSeq)

aSeq = aSeq.astype('uint32')

df = df.assign(SeqID = aSeq)


df.dtypes

df.columns = ['MD', 'D', 'B', 'S', 'ID', 'F0', 'F1', 'T0']

df = df.reset_index()

df = df.drop(['SeqID'], axis = 1)


df = df[['ID', 'B', 'S', 'F0', 'F1', 'T0', 'MD', 'D']]

df = df.sort_values(by = ['ID', 'F0', 'F1', 'T0'])

df.to_pickle(DATA_DIR + "noised_30JUL2018_cleaned.pkl")

#####################################################################################



