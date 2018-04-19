

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
    return 3.7 * X[:,5] + 0.1 * X[:,1] + 2

"""c"""

# Linear 
#
# Set up model, with secret function
#
# f = c0*x0 + c1*x1 + c2*x2 + c3*x3 + c4*x4 + c5*x5
#

def get_init_X():

    nRows = 1005

    x_in_0 = np.random.uniform(size= nRows)
    x_in_1 = np.random.uniform(size= nRows)
    x_in_2 = np.random.uniform(size= nRows)

    l = []

    l.append (x_in_0 * x_in_0 * x_in_0)   # 0
    l.append (x_in_1 * x_in_1 * x_in_1)   # 1 
    l.append (x_in_2 * x_in_2 * x_in_2)   # 2

    l.append (x_in_0 * x_in_1 * x_in_2)   # 3

    l.append (np.log(x_in_0) * np.log(x_in_1) * np.log(x_in_2))   # 4

    l.append (x_in_0 * np.log(x_in_0) * x_in_1 * np.log(x_in_1) *  x_in_2 *np.log(x_in_2) )  # 5

    l.append (np.ones(nRows))

    X = np.stack(l, axis = 1)

    return X

"""c"""


X = get_init_X()

y_true = get_true_value(X)


d = WeightDeterminator_GetWeigths(X, y_true)









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

