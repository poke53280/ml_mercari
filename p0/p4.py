

###############################################################################
#
# Analysis - timings with TimeLineTool.py
#


import pandas as pd
import numpy as np
import more_itertools as mit

import sys
import os

import numpy as np

sys.path

cwd = os.getcwd()

os.chdir('C:\\Users\\T149900\\ml_mercari')

from general.TimeLineTool import *


DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


df = pd.read_pickle(DATA_DIR + "noised_30JUL2018_cleaned.pkl")

df.shape[0]

df = df.drop_duplicates()

df.shape[0]

df.dtypes


t_start = 14000
t_end = 17000



################################################################################
#
#  GetTargetIntervals - example
#
# Setup:


df_m = pd.DataFrame()

df_m['IDX'] = df.ID  #Individual ID.
df_m['F1'] =  df.F1  #From day inclusive
df_m['Q'] =   df.T0  #To day inclusive

m = (df_m.Q < df_m.F1)

assert len (df[m]) == 0, "Invalid intervals given"

# t_start, t_end defined a time line interval

min_t = np.min(df_m.F1)
max_t = np.max(df_m.Q)

L = max_t - min_t

rGrow = .1

min_t = int(min_t - rGrow * L)
max_t =int (max_t + rGrow * L)

t_start = min_t
t_end = max_t





l = GetTargetInterval(df_m, min_t, max_t)


l[89]
(31220, 31455, 1)

q = df_m.IDX == 89
df[q]



