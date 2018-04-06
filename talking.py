

import pandas as pd
import numpy as np

DATA_DIR_PORTABLE = "C:\\Users\\T149900\\ml_mercari\\"
DATA_DIR_BASEMENT = "c:\\data_talking\\"
DATA_DIR = DATA_DIR_BASEMENT


#print('loading train data...')
#df = pd.read_csv(DATA_DIR + "train_sample.csv")


from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error


def mae_func(weights, predictions, y_true):

    sum = 0

    for i in range(len(predictions)):
        sum +=  weights[i] * predictions[i]

    return mean_absolute_error(y_true, sum)

"""c"""

nRows = 200000

mu, sigma = 0, 0.1

y_true = 10 + 1.1 * np.arange(nRows) + np.random.normal(mu + 0, sigma * 0.3, nRows)

predictions = []

predictions.append(1.2 * y_true + 9 + np.random.normal(mu + 0, sigma * 1, nRows))
predictions.append(0.9 * y_true + 4 + np.random.normal(mu + 0.3, sigma * 2, nRows))
predictions.append(1.1 * y_true + 3 + np.random.normal(mu - 0.3, sigma * 3, nRows))
predictions.append(0.9 * y_true - 1  + 2 * np.random.normal(mu + 0, sigma * 4, nRows))
predictions.append(0.9 * y_true + 4 + 3* np.random.normal(mu + 0.3, sigma * 2, nRows))
predictions.append(1.05 * y_true + 1 + 4* np.random.normal(mu - 0.5, sigma * 3, nRows))
predictions.append(0.95 * y_true - 3  + 3* np.random.normal(mu + 0, sigma * 6, nRows))
predictions.append(1.0 * np.ones(nRows))





def GetWeigths(predictions, y_true):

    nPredictions = len (predictions)

    d = {}
    w_out = {}

    # Single best

    print("Single...")

    w = np.zeros(nPredictions)
    w[0] = 1

    min = 100000

    for x in range(len(predictions)):
        a = np.roll(w, x)
        loss = mae_func(a, predictions, y_true)
        min = np.minimum(min, loss)

    """c"""

    d['single'] = min

    # Mean

    print("Mean...")

    w0 = np.ones(nPredictions)/nPredictions
    mean = mae_func(w0, predictions, y_true)

    d['mean'] = mean

    print("L-BFGS-B...")

    w0 = np.ones(nPredictions)/nPredictions
    res = minimize(mae_func, w0, (predictions, y_true), method='L-BFGS-B')
    loss = mae_func(res.x, predictions, y_true)

    d['L-BFGS-B'] = loss
    w_out['L-BFGS-B'] = res.x

    print("SLSQP...")
    w0 = np.ones(nPredictions)/nPredictions
    res = minimize(mae_func, w0, (predictions, y_true), method='SLSQP')
    loss = mae_func(res.x, predictions, y_true)

    d['SLSQP'] = loss
    w_out['SLSQP'] = res.x

    print("Best of 100...")

    lls_A= []
    wghts_A = []

    lls_B = []
    wghts_B = []

    lls_C= []
    wghts_C = []

    for i in range(100):
    
        starting_values = np.random.uniform(size=len(predictions))

        res = minimize(mae_func, starting_values, (predictions, y_true), bounds=  [(0,1)]*len(predictions), method='SLSQP', options={'disp': False, 'maxiter': 50})

        f = res['fun']
        w = res['x']

        lls_A.append(f)
        wghts_A.append(w)

        if np.min(lls_A) == f:
            print(f"[{i}]0_1: {f}")

        res = minimize(mae_func, starting_values, (predictions, y_true), method='SLSQP', options={'disp': False, 'maxiter': 50})

        f = res['fun']
        w = res['x']

        lls_B.append(f)
        wghts_B.append(w)

        if np.min(lls_B) == f:
            print(f"[{i}]unbound: {f}")

        res = minimize(mae_func, starting_values, (predictions, y_true),  bounds=  [(-1.2,1.2)]*len(predictions), method='SLSQP', options={'disp': False, 'maxiter': 50})

        f = res['fun']
        w = res['x']

        lls_C.append(f)
        wghts_C.append(w)

        if np.min(lls_C) == f:
            print(f"[{i}]1_2: {f}")


    """c"""

    f_A = np.min(lls_A)
    w_A = wghts_A[np.argmin(lls_A)]

    f_B = np.min(lls_B)
    w_B = wghts_B[np.argmin(lls_B)]

    f_C = np.min(lls_C)
    w_C = wghts_C[np.argmin(lls_C)]

    d['b100_0_1'] = f_A
    w_out['b100_0_1'] = w_A

    d['b100_unbound'] = f_B
    w_out['b100_unbound'] = w_B

    d['b100_1.2'] = f_C
    w_out['b100_1.2'] = w_C

    return d, w_out

"""c"""

