

#######################################
#
#  "5th Place Solution of Favorita Grocery Sales Forecasting"
#
#

#http://forums.fast.ai/t/5th-place-solution-of-favorita-grocery-sales-forecasting/10009


#https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47556

#
# Code
#https://github.com/LenzDu/Kaggle-Competition-Favorita

import time
import pandas as pd
import numpy as np
import gc
from category_encoders import *
  
DATA_DIR_PORTABLE = "C:\\favorita_data\\"  
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

def load_frame(name):
    print(f"Loading {name}...")
    return pd.read_table(DATA_DIR + name + ".csv", sep = ",", low_memory=False);

"""c"""

df_i = load_frame("items")

df_tr = load_frame("train")


name = "train"

df = pd.read_csv(DATA_DIR + name + ".csv", sep = ",", nrows = 12, parse_dates=["date"])

#
#  Indexing tutorial
#
# https://medium.com/@shirleyliu/pandas-101-indexing-5a88e2c72f9f
#
#

s0 = pd.Series(index = [9,8,78,19, 33], data = [1,2,3,4, 11])
s1 = pd.Series(index = [17,1,7,29, 33], data = [9,10,11,12, 15])


df = pd.DataFrame({'col1':s0, 'col2':s1})

df.reindex([9, 17, 33, 1])



m = s.index.duplicated()

isExistDuplicates = (m == True).any()

assert (not isExistDuplicates)


s = s.reindex([8,7,3,4,8,8, 78])




s.reset_index(drop = True, inplace=True)






df = pd.DataFrame(data = {'age': [10,28,30], 'weight': [120,133,155],'height': [160,165,175], 'color':['blue','green','pink']}, columns = ["age","weight", "height","color"])


