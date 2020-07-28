
#
#
# https://blog.tensorflow.org/2018/12/an-introduction-to-probabilistic.html
#
#

import numpy as np
import matplotlib.pyplot as plt


def logistic_function(t, alpha, beta):
    return 1.0 / ( 1.0 + np.exp(beta * t + alpha))


l_t = np.arange(-10, 10, 0.1)
l_p = [logistic_function(t, 13.3, 3.2) for t in l_t]

plt.plot(l_t, l_p)

plt.show()



