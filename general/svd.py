

import numpy as np

LA = np.linalg

a = np.array([[1, 3, 4], [5, 6, 9], [1, 2, 3], [7, 6, 8]])
print(a)

U, s, Vh = LA.svd(a, full_matrices=False)

assert np.allclose(a, np.dot(U, np.dot(np.diag(s), Vh)))


s
np.diag(s)
np.dot(Vh[1],Vh[2])

U