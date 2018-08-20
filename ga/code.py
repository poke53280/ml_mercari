
#
# DEAP - sklearn
#

import numpy as np

from evolutionary_search import maximize
from evolutionary_search import minimize

def func(x, y, m=1., z=False):
    return m * (np.exp(-(x**2 + y**2)) + float(z))

param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}

args = {'m': 1.}

best_params, best_score, score_results, _, _ = maximize(func, param_grid, args, verbose=True)



w0 = np.ones(nFeatures)/nFeatures

param_grid = {}

for w in range(nFeatures):
    key = 'w' + str(w)
    print (key)

    param_grid[key] = [-10, 0, 10]


minimize(error_func, param_grid, (X, y_true))


# From WeightDeterminator.py: res = minimize(error_func, w0, (X, y_true), method='L-BFGS-B')


