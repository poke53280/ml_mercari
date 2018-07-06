
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

def swap_rows(X_batch, X_clean, p):

    X_out = X_batch.copy()

    #nCols = X.shape[1]
    nRowsBatch = X_batch.shape[0]

    nRowsClean = X_clean.shape[0]

    for iRow in range(nRowsBatch):

        target = np.squeeze(np.asarray( X_batch[iRow, :]))

        iNoiseRow = np.random.choice(range(nRowsClean))

        noise = np.squeeze(np.asarray( X_clean[iNoiseRow, :]))

        res = _get_noised_line (target, noise, p)
        X_out[iRow] = res

    return X_out

"""c"""

def test():

    X_clean =[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [47, 47, 47, 47, 47, 47, 47, 47, 47, 47],
        [11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        [81, 81, 81, 81, 81, 81, 81, 81, 81, 81],
        [21, 21, 21, 21, 21, 21, 21, 21, 21, 21],
        [19, 19, 19, 19, 19, 19, 19, 19, 19, 19]]

    X_clean = np.matrix(X_clean)

    X_clean.shape

    X_batch = [[99,99,99,99,99,99,99,99,99,99], [77,77,77,77,77,77,77,77,77,77], [55,55,55,55,55,55,55,55,55,55]]

    X_batch = np.matrix(X_batch)

    X_res = swap_rows(X_batch, X_clean, 0.3)

    print(X_batch)
    print(X_res)

"""c"""



