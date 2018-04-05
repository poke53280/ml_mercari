

import pandas as pd
import numpy as np

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
DATA_DIR_BASEMENT = "c:\\data_talking\\"
DATA_DIR = DATA_DIR_BASEMENT


print('loading train data...')
df = pd.read_csv(DATA_DIR + "train_sample.csv")


from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error


def mae_func(weights, predictions, y_c):

    sum = 0

    for i in range(len(predictions)):
        sum +=  weights[i] * predictions[i]

    return mean_absolute_error(y_c, sum)

"""c"""


y_c = np.array([10.0, 20.0, 30.0, 40.0, 50.05, 111.0])
y_1 = np.array([0.9, 1.9, 2.8, 3.9, 5.0,  7.3])
y_2 = np.array([12.1, 2.1, 3.1, 4.2, 5.1,  14.9])
y_3 = np.array([0.8, 1.8, 2.7, 3.8, 4.9,  10.0])

predictions = []

predictions.append(y_1)
predictions.append(y_2)
predictions.append(y_3)

nPredictions = len (predictions)

for y in predictions:
    mean_absolute_error(y_c, y)

"""c"""

w0 = np.ones(nPredictions)/nPredictions

bnds = tuple((0,1) for w in w0)

bnds = [(-0.1,100)]*len(predictions)

cons = ({"type":"eq","fun":lambda w: 1-w.sum()})

res = minimize(mae_func, w0, (predictions, y_c), method='L-BFGS-B', bounds = bnds ,  options={'disp': False, 'maxiter': 100000})


res.x


mae_func(res.x, predictions, y_c)


res.x