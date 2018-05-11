
# Linear regression function
#
# Driver further below.


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

#################################################################
#
#  WeightDeterminator_Get_X
#

def WeightDeterminator_Get_X(x0, x1, x2):

    assert (x0.shape[0] == x1.shape[0])
    assert (x0.shape[0] == x2.shape[0])

    nRows = x0.shape[0]

    l = []

    s = []

    # Linear in one
    l.append(x0)
    s.append("x0")

    l.append(x1)
    s.append("x1")

    l.append(x2)
    s.append("x2")

    x_mul_0_1 = x0 * x1
    x_mul_0_2 = x0 * x2
    x_mul_1_2 = x1 * x2

    # Linear in two
    l.append(x_mul_0_1)
    s.append("x0 * x1")
    
    l.append(x_mul_0_2)
    s.append("x0 * x2")

    l.append(x_mul_1_2)
    s.append("x1 * x2")

    x_sqr_0 = x0 * x0
    x_sqr_1 = x1 * x1
    x_sqr_2 = x2 * x2

    #Quadratic in one

    l.append(x_sqr_0)
    s.append("x0 * x0")

    l.append(x_sqr_1)
    s.append("x1 * x1")

    l.append(x_sqr_2)
    s.append("x2 * x2")

    # Quadratic in 1, linear in the other two:

    l.append (x_sqr_0 * x_mul_1_2)
    s.append("x0 * x0 * x1 * x2")

    l.append (x_sqr_1 * x_mul_0_2)
    s.append("x1 * x1 * x0 * x2")

    l.append (x_sqr_2 * x_mul_0_1)
    s.append("x2 * x2 * x0 * x1")

    # Linear in all:
    l.append (x0 * x1 * x2)   
    s.append("x0 * x1 * x2")

    # Linear in log all:
    l.append (np.log(x0) * np.log(x1) * np.log(x2))   
    s.append("log(x0) * log(x1) * log(x2)")


    # Linear in xlog(x) all
    l.append (x0 * np.log(x0) * x1 * np.log(x1) *  x2 *np.log(x2) )  
    s.append("x0 * log(x0) * x1 * log(x1) *  x2 * log(x2)")

    #Quadratic in all
    l.append (x_sqr_0 * x_sqr_1 * x_sqr_2)
    s.append("x0 * x0 * x1 * x1 * x2 * x2")

    l.append (np.ones(nRows))
    s.append("const")

    X = np.stack(l, axis = 1)

    return X, s

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



