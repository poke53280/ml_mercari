
from scipy.sparse import csr_matrix
import numpy as np


###########################################################################################
#
#         GetCSR_X_Variance
#
#   Var(X) = Mean[X^2] - (Mean[X])^2

def GetCSR_X_Variance(X):

    squared_X = X.copy()
    squared_X.data **= 2

    m = X.mean(axis = 0)

    m2 = squared_X.mean(axis = 0)

    m = np.squeeze(np.asarray(m))

    m2 = np.squeeze(np.asarray(m2))

    m = (m * m)
    var = m2 - m

    return var

"""c"""

#############################################################################
#
#           GetUniqueRows()
#
#  http://www.ryanhmckenna.com/2017/01/efficiently-remove-duplicate-rows-from.html
#
# A = A[idx]

def GetUniqueRows(A):
    x = np.random.rand(A.shape[1])
    y = A.dot(x)
    unique, index = np.unique(y, return_index=True)
    return index

"""c"""

#####################################################################################
#
#           GetUniqueColumns()
#
# Just by transposing and calling GetUniqueRows(). Can reduce memory consumption.
#
#
#  A = A[:, idx]
#

def GetUniqueColumns(A):
    At = A.copy()
    At = At.T
    return GetUniqueRows(At)

"""c"""




