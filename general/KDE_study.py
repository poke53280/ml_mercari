

#
# https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
#



import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

###############################################################################
#
#           group_unique_integers_around_mean
#
# Input integer array with unique elements
#
# Return interval centered around mean with same amount of elements.
#

def group_unique_integers_around_mean(a):

    assert len(np.unique(a)) == len (a)

    nElements = len(a)

    # Rasterize
    # Center at mean:
    c = a.mean()

    lo = c - (nElements/2)
    
    # Round down if there is element there.
    if int(lo) == np.min(a):
        lo = np.min(a)
    else:
        lo = int (lo + .5)

    # Returns both ends inclusive
    return (lo, lo + nElements - 1)

###############################################################################
#
#           group_unique_integers_from_min
#
# Input integer array with unique elements
#
# Return interval starting from smallest element.
#

def group_unique_integers_from_min(a):

    assert len(np.unique(a)) == len (a)

    nElements = len(a)

    lo = np.min(a)

    # Returns both ends inclusive
    return (lo, lo + nElements - 1)

"""c"""

###############################################################################
#
#           get_minima
#
#

def get_minima(a, bandwidth, isPlot):

    a_low = a.min() - 2
    a_hi = a.max() + 2

    a = a.reshape(-1, 1)

    num_x = 300 + a_hi - a_low

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(a)

    s = np.linspace(a_low, a_hi, num = num_x)

    s = s.reshape(-1, 1)

    e = kde.score_samples(s)

    e = np.exp(e)


    # Fill in low values on -inf for a derivable function
    #
    # e_min = np.min(e[np.isfinite(e)])
    # e [np.isneginf(e) ] = (e_min - 1)

    if isPlot:
        plt.plot(s, e)
        plt.show()

    mi = argrelextrema(e, np.less)[0]

    s = s.squeeze()

    minima = s[mi]

    nMinima = len (minima)

    l = []

    if nMinima == 0:
        # print("No minima found, no split")
        l.append(a)

    else:
        l.append(a[a < minima[0]])

        for i in range(0, nMinima -1):
            l.append (a[ (a > minima[i]) & (a < minima[i+1])])

        """c"""

        l.append(a[a > minima[nMinima -1]])

    return l

###############################################################################
#
#           group_sorted_unique_integers
#
#

def group_sorted_unique_integers(a, bandwidth, isPlot):
    
    # Ensure unique and sorted
    assert len(np.unique(a)) == len(a)
    assert (a == np.sort(a)).all()

    l = get_minima(a, bandwidth, isPlot)

    clustered = []

    for x in l:
        ang = np.array(x)

        if len(ang) == 0:
            pass
        else:
            clustered.append(group_unique_integers_from_min(ang))

    """c"""
        
    return clustered

"""c"""



########################################## MAIN #############################################


def test_example():

    # Input data
    n_bandwidth = 3

    # Input sorted integers, no duplicates

    a = np.array([ 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 17, 18, 22, 23, 24, 29, 30, 31]) 

    l = group_sorted_unique_integers(a, n_bandwidth, True)


    a = np.array(range(9, 3000))
    b = np.array(range(3050, 6000))

    c = np.concatenate([a, b])

    l = group_sorted_unique_integers(a, n_bandwidth, True)

    

