
########################################################################
#
# Testing merge and group
#


import pandas as pd
import numpy as np


DATA_DIR_PORTABLE = "C:\\Users\\T149900\\data_merge_tutorial\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

def load_frame(name):
    print(f"Loading {name}...")
    return pd.read_table(DATA_DIR + name + ".csv", sep = ",");

"""c"""


user_usage = load_frame("user_usage")
user_device = load_frame("user_device")
devices = load_frame("android_devices")

result = pd.merge(user_usage, user_device[['use_id', 'platform', 'device']], on= 'use_id', how= 'left')

result.shape

#
# All on one line => group by
#
# ref home_credit_main:
#

q = d['bb']

m = (q.SK_ID_BUREAU == 5041336) | (q.SK_ID_BUREAU == 5041332)

q = q[m]


"""c"""

# https://stackoverflow.com/questions/22219004/grouping-rows-in-list-in-pandas-groupby

df = pd.DataFrame( {'a':np.random.randint(0,60,600), 'b':[1,2,5,5,4,6]*100})

keys, values = df.sort_values('a').values.T

ukeys,index=np.unique(keys, return_index = True)

arrays=np.split(values,index[1:])

df2=pd.DataFrame({'a':ukeys,'b':[list(a) for a in arrays], 'c':[my_procedural_array(a) for a in arrays]})

def my_procedural_array(a):
    return [9, sum(a)]

"""c"""


g = q.groupby(by = 'SK_ID_BUREAU')


for m, s in g:
    print(f"balance = {m}, status = {s}")
