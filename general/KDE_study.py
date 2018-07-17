

#
# https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
#


import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


def sort_to_cluster(a, c):
    assert (a == np.sort(a)).all()
    assert (c == np.sort(c)).all()

    d = {}

    for x in range(len(c)):
        d[x] = []

    for value in a:
        idx = (np.abs(c - value)).argmin()
        d[idx].append(value)

    return d

###############################################################################
#
#           group_unique_integers
#
# Input integer array with unique elements
#
# Return interval centered around mean with same amount of elements.
#

def group_unique_integers(a):

    assert len(np.unique(a)) == len (a)

    nElements = len(a)

    # Rasterize
    # Center at mean:
    c = a.mean()

    c = int (c + 0.5)

    lo = c - (nElements -1)// 2

    # Returns both ends inclusive
    return (lo, lo + nElements - 1)


###############################################################################
#
#           group_sorted_unique_integers
#
#

def group_sorted_unique_integers(a, bandwidth):
    
    # Ensure unique and sorted
    assert len(np.unique(a)) == len(a)
    assert (a == np.sort(a)).all()

    a_low = a.min() - 2
    a_hi = a.max() + 2

    a = a.reshape(-1, 1)

    # A few more than the integer range
    num_x = 3 + a_hi - a_low

    kde = KernelDensity(kernel='linear', bandwidth=bandwidth).fit(a)

    s = np.linspace(a_low, a_hi, num = num_x)

    s = s.reshape(-1, 1)

    e = kde.score_samples(s)

    plt.plot(s, e)
    plt.show()

    ma = argrelextrema(e, np.greater)[0]

    print(f"Suggesting {len(ma)} group(s)")

    s = s.squeeze()

    maxima = s[ma]

    print(f"Maxima: {s[ma]}")

    c = s[ma]

    a = a.squeeze()

    d = sort_to_cluster(a, c)

    l = []

    for idx, g in d.items():
        ang = np.array(g)
        l.append(group_unique_integers(ang))
        
    return l


"""c"""



########################################## MAIN #############################################


def test_example():

    # Input data
    n_bandwidth = 3

    # Input sorted integers, no duplicates

    a = np.array([ 9, 11, 12, 14, 22, 24, 25, 27, 29, 34, 35, 38, 39, 50])

    l = group_sorted_unique_integers(a, n_bandwidth)





