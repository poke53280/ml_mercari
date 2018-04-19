

import pandas as pd
import numpy as np
import datetime
import pylab as pl
import gc

from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize


#################################################################
#
#  get_true_value
#

def get_true_value(X):
    x0 = X[:,0]
    x1 = X[:,1]
    x2 = X[:,2]

    return 3.7 * x0 * x1 * x2 + 1.9 * x0 * x2 + 0.7

"""c"""



nRows = 1005

# The three independent variables

x0 = np.random.uniform(size= nRows)
x1 = np.random.uniform(size= nRows)
x2 = np.random.uniform(size= nRows)

X, s = WeightDeterminator_Get_X(x0, x1, x2)

y_true = get_true_value(X)

d = WeightDeterminator_GetWeigths(X, y_true)

w = d[1]['SLSQP']

for i, x in enumerate(w):
    if np.abs(x) > 0.01:
        print (f"{s[i]} w = {x}")

"""c"""





DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\synthpop-data\\"            
DATA_DIR_BASEMENT = "c:\\data_talking\\"
DATA_DIR = DATA_DIR_PORTABLE


print('loading train data...')

df = pd.read_csv(DATA_DIR + "synthpop_timing.csv")

df = df.drop(['Unnamed: 0'], axis = 1)


def get_X(w, col):

    n = col[0]
    a = col[1]
    b = col[2]

    c0 = w[0]
    c1 = w[1]
    c2 = w[2]
    c3 = w[3]

    s = c0 * np.log(n) * np.log(a) * np.log(b)
    return s


def mae_func(w, col, t_true):

    t_p = get_X(w, col)

    return mean_absolute_error(t_true, t_p)

"""c"""


n = np.array(df.n)
a = np.array(df.a)
b = np.array(df.b)
t = np.array(df.t)


col = [n, a, b]
t_true = t


w0 = np.random.uniform(size=4)

res = minimize(mae_func, w0, (col, t_true), method='SLSQP', options={'disp': False, 'maxiter': 50})

res['fun']




t_p = get_X(res.x, col)


res.x

