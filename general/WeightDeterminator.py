
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error


def mae_func(weights, predictions, y_true):

    sum = 0

    for i in range(len(predictions)):
        sum +=  weights[i] * np.array(predictions[i])

    err = mean_absolute_error(y_true, sum)

    return err

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

#################################################################
#
#  get_predict_value
#

def get_predict_value(w, X):

    X_p = np.multiply(X, w)

    y_p = X_p.sum(axis = 1)

    return y_p

"""c"""


##################################################
#
#  error_func
#

def error_func(w, X, y_t):
  
    y_p = get_predict_value(w, X)

    return mean_absolute_error(y_t, y_p)

"""c"""


##################################################
#
#  WeightDeterminator_GetWeigths
#

def WeightDeterminator_GetWeigths(X, y_true):

    nFeatures = X.shape[1]

    d = {}
    w_out = {}

    # Single best

    print("Single...")

    w = np.zeros(nFeatures)
    w[0] = 1

    min = 100000

    for x in range(nFeatures):
        a = np.roll(w, x)
        loss = error_func(a, X, y_true)
        min = np.minimum(min, loss)

    """c"""

    d['single'] = min

    # Mean

    print("Mean...")

    w0 = np.ones(nFeatures)/nFeatures
    mean = error_func(w0, X, y_true)

    d['mean'] = mean

    print("L-BFGS-B...")

    w0 = np.ones(nFeatures)/nFeatures
    res = minimize(error_func, w0, (X, y_true), method='L-BFGS-B')
    loss = error_func(res.x, X, y_true)

    d['L-BFGS-B'] = loss
    w_out['L-BFGS-B'] = res.x

    print("SLSQP...")
    w0 = np.ones(nFeatures)/nFeatures
    res = minimize(error_func, w0, (X, y_true), method='SLSQP')
    loss = error_func(res.x, X, y_true)

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
    
        starting_values = np.random.uniform(size=nFeatures)

        res = minimize(error_func, starting_values, (X, y_true), bounds = [(0,1)]*nFeatures, method='SLSQP', options={'disp': False, 'maxiter': 50})

        f = res['fun']
        w = res['x']

        lls_A.append(f)
        wghts_A.append(w)

        if np.min(lls_A) == f:
            print(f"[{i}]0_1: {f}")

        res = minimize(error_func, starting_values, (X, y_true), method='SLSQP', options={'disp': False, 'maxiter': 50})

        f = res['fun']
        w = res['x']

        lls_B.append(f)
        wghts_B.append(w)

        if np.min(lls_B) == f:
            print(f"[{i}]unbound: {f}")

        res = minimize(error_func, starting_values, (X, y_true),  bounds= [(-1.2,1.2)]*nFeatures, method='SLSQP', options={'disp': False, 'maxiter': 50})

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




