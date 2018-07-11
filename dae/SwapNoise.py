
import numpy as np

def add_swap_noise(X_batch, X_clean, p):

    nNumRowsBatch = X_batch.shape[0]
    nNumRowsSource = X_clean.shape[0]

    print(f"Adding {p * 100.0}% noise to {nNumRowsBatch} row(s) from noise pool of {nNumRowsSource} row(s).")

    aiNoiseIndex = np.random.randint(nNumRowsSource, size=nNumRowsBatch)

    X_noise = X_clean[aiNoiseIndex]

    X_mask = np.random.rand(X_batch.shape[0], X_batch.shape[1])

    m = X_mask < p

    X_batch[m] = 0
    X_noise[~m] = 0

    X_batch = X_noise + X_batch

    return X_batch
"""c"""

def swap_test():

    X_clean_const =[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [47, 47, 47, 47, 47, 47, 47, 47, 47, 47],
        [11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
        [81, 81, 81, 81, 81, 81, 81, 81, 81, 81],
        [21, 21, 21, 21, 21, 21, 21, 21, 21, 21],
        [19, 19, 19, 19, 19, 19, 19, 19, 19, 19]]

    X_batch = [[99,99,99,99,99,99,99,99,99,99], [77,77,77,77,77,77,77,77,77,77], [55,55,55,55,55,55,55,55,55,55]]

    X_clean_const = np.array(X_clean_const)

    X_batch = np.array(X_batch)

    X_batch = add_swap_noise(X_batch, X_clean_const, 0.15)