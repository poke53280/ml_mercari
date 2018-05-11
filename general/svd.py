

import numpy as np

def IsOrthogonal(A):
    I = np.zeros(A.shape, dtype = float)
    np.fill_diagonal(I, 1)

    return np.isclose(np.dot(A, A.T), I).all()

"""c"""

LA = np.linalg

a = np.array([[1, 0, 0, 0, 2],
              [0, 0, 3, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 2, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 2, 0, 0, 0]])


print(a)

U, s, Vh = LA.svd(a, full_matrices=True)

# Vh: Initial rotation
# s : Scaling along the coordinate axes
# U : Final rotation
#
# Holds with full_matrices False: assert np.allclose(a, np.dot(U, np.dot(np.diag(s), Vh)))
#

np.dot(U, U.T)


np.dot(U[0], U[3])

np.dot(Vh[1],Vh[2])

a.shape
U.shape
s.shape
Vh.shape


np.allclose(a, np.dot(U, np.dot(s, Vh)))


