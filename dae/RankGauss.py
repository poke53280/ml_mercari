
##########################################
#
# From https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629 
#
# by Tom - https://www.kaggle.com/studynon
#
#

import numpy as np
import pandas as pd
from scipy.special import erfinv
import matplotlib.pyplot as plt

######################################################################
#
#      rank_gauss
#
#

def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x


######################################################################
#
#      rank_gauss_test
#
#
# x = np.random.rand(500)

def rank_gauss_test(x):

    
    # histogram test: the histogram of rank_gauss should be gauss-liked and centered
    pd.Series(x).hist()
    plt.show()
    pd.Series(rank_gauss(x)).hist()
    plt.show()

"""c"""

######################################################################
#
#      gauss_rank_transform
#

def gauss_rank_transform(x):
    return pd.Series(rank_gauss(x))


######################################################################
#
#      test_gauss_rank_transform
#

def test_gauss_rank_transform():
    q = train_CONST.apply(gauss_rank_transform, axis = 0, raw = True)







