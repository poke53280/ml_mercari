

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
from scipy.stats import norm

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 900)
pd.set_option('display.max_colwidth', 500)


x = np.array([3,4,1,7, 16, 29, 70], dtype = np.int32)

nSpace = 10

x_d = np.arange(np.min(x) - nSpace, np.max(x) + nSpace + 1, dtype = np.float32)

support = np.linspace(np.min(x) - nSpace, np.max(x) + nSpace +1, 100)

x = x.reshape(-1, 1)

variance = 4

a_ = norm(x, variance).pdf(support)

for a_row in a_:
    plt.plot(support, a_row, color="r")

"""c"""

sns.rugplot(x, color=".2", linewidth=3)

density = np.sum(a_, axis = 0)

plt.plot(support, density)


plt.show()


a_ = norm(x, variance).pdf(x_d)

density = np.sum(a_, axis = 0)


m = density > 0.001

x_s = x_d[m]


i_start = np.where(np.diff(x_s) > 1)[0]
i_start = i_start + 1
i_start = np.insert(i_start, 0, 0)

i_end = i_start[1:]
i_end = np.append(i_end, x_s.shape[0])

x_lo = x_s[i_start]
x_hi = x_s[i_end -1]




