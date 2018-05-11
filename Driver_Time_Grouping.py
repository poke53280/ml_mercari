
import numpy as np
import matplotlib.pyplot as plt

import general.TimeLineTool as tl




v = df.ip_app_sys_channel.value_counts()

idx = v.index
count = v.values

tup = zip(idx, count)

opt = []
sum_length = 0


cumsum = np.cumsum(count)
sum = np.sum(count)

cumsum = cumsum/sum

len(idx)
66451050

test_idx_into_list = int(66451050/950)

cumsum[test_idx_into_list]

count[test_idx_into_list]

test_idx = idx[test_idx_into_list]


for idx, c in tup:
    if c > 5000:
        print(f"idx = {idx}, count = {c}")
        m = df.ip_app_sys_channel == idx
        q = df[m]
        print(f"{idx} : length = {len (q)}")

        acData = q.time.values
        acData = np.sort(acData)

        optimal = TimeLineTool_GetOptimalGroupSize(acData, False, 5 * 60)

        opt.append(optimal)

        sum_length += len (q)

        factor = sum_length / len(df)
        print(f"factor is {factor*100:.1f}%")



"""c"""


l = [29129833, 29152905, 29135705, 29158771, 29152899, 29129827, 29158765, 29135699, 2198559, 2198553, 60230323, 2158031, 2209611]

opt = []



for comboID in l:
    m = df.ip_app_sys_channel == comboID
    q = df[m]
    print(f"{comboID} : length = {len (q)}")

    acData = q.time.values
    acData = np.sort(acData)

    optimal = TimeLineTool_GetOptimalGroupSize(acData, False, 5 * 60)

    opt.append(optimal)

    sum_length += len (q)

"""c"""

# from count 100 (6940 items)

m = df.ip_app_sys_channel == 57219209

q = df[m]

print(f"{len(q)}")

# seven values, very far away from each other.
# => seven groups of max density



acData = q.time.values
acData = np.sort(acData)

optimal = TimeLineTool_GetOptimalGroupSize(acData, True, 12* 3600)


TimeLineTool_Analyze_Cluster(acData, 70)

#
# Isolate single users by grouping, clustering.
#
# Gives:
#
# u0       c.....c.....c.........c.........c....c.......
# u1       c......X.......c.......c.......X........c....
# u2       .....c............c........X........c........
#
#
# Optimally insert Xs in test set, based on c
#
#  u0       c.....c.....c.........c.........c....c.......
#  u1       c.............c.......c................c.....
#  u2       .....c............c.................c........
#
# Cluster test set to train set. Pick test users closest to convert users in train.
#
#

u0_c = [223, 400, 500, 600]
u0_a = [450]




#
# https://stackoverflow.com/questions/35094454/how-would-one-use-kernel-density-estimation-as-a-1d-clustering-method-in-scikit
#
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
#
#

from numpy import array, linspace
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt


a = df1.time.values.reshape(-1,1)

a = array([10,11,9,23,21,11,45,20,11,12]).reshape(-1, 1)

a = acData.reshape(-1, 1)

kde = KernelDensity(kernel='linear', bandwidth=3).fit(a)

s = linspace(0, 220000)

e = kde.score_samples(s.reshape(-1,1))

plot(s, e)

plt.show()

from scipy.signal import argrelextrema

mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

print(f"Minima: {s[mi]}")
print(f"Maxima: {s[ma]}")

m = a < mi[0]
a[m]

m = a >= mi[0] & a < mi[1]
a[m]

m = (a >= mi[1])
a[m]

plot(s[:mi[0]+1], e[:mi[0]+1], 'r',
     s[mi[0]:mi[1]+1], e[mi[0]:mi[1]+1], 'g',
     s[mi[1]:], e[mi[1]:], 'b',
     s[ma], e[ma], 'go',
     s[mi], e[mi], 'ro')







