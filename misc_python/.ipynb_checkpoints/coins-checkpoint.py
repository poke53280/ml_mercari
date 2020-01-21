
import numpy as np
import pandas as pd
from scipy.stats import bernoulli


n, p = 1, .33  # n = coins flipped, p = prob of success

s = np.random.binomial(3000, 0.50)

from sympy.stats import Bernoulli, sample_iter
list(sample_iter(Bernoulli('X', 0.8), numsamples=10)) # p = 0.8 and nsamples=10


