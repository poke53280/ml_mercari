
import numpy as np

#########################################################################
#
#           get_noised_line
#

def _get_noised_line(init, noise, p):
    N = len (init)

    ar = np.random.rand(N)
    ab = (ar < p)

    res = np.empty(N, dtype = init.dtype)

    for ix, (init_value, noise_value, is_swap) in enumerate(zip (init, noise, ab)):
        res[ix] = noise_value if is_swap else init_value

    return res

"""c"""

#########################################################################
#
#           swap_rows
#

def swap_rows(X, p):

    X_out = X.copy()

    nCols = X.shape[1]
    nRows = X.shape[0]

    for iRow in range(nRows):

        if iRow % 1000 == 0:
            print(f"Processing row {iRow}...")

        target = np.squeeze(np.asarray( X[iRow, :]))

        iNoiseRow = np.random.choice(range(nRows))

        noise = np.squeeze(np.asarray( X[iNoiseRow, :]))

        res = _get_noised_line (target, noise, p)
        X_out[iRow] = res

    return X_out

"""c"""

def test():

    X = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [47, 47, 47, 47, 47, 47, 47, 47, 47, 47],
        [11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        [81, 81, 81, 81, 81, 81, 81, 81, 81, 81],
        [21, 21, 21, 21, 21, 21, 21, 21, 21, 21],
        [19, 19, 19, 19, 19, 19, 19, 19, 19, 19],
        [29, 29, 29, 29, 29, 29, 29, 29, 29, 29]]

    X = np.matrix(X)

    X.shape

    X_res = swap_rows(X, 0.1)

    print(X)
    print(X_res)

"""c"""



