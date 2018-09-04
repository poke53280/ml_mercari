

###############################################################################
#
#       Analysis - timings with TimeLineTool.py
#
#

import pandas as pd
import numpy as np
import more_itertools as mit
import sys
import os
import numpy as np
from scipy.stats import skew, kurtosis

sys.path
cwd = os.getcwd()
os.chdir('C:\\Users\\T149900\\ml_mercari')




"""c"""





t_start = 20000
t_end = 37000
nGrow = 15


nAllIDS = len (np.unique(df.ID))

nCut = nAllIDS  #  = nAllIDS for no cut

# First cut historic data 
m = (df.ID < nCut)
df = df[m]





