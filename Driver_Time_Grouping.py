
import numpy as np



###############################################################
#
#
#

def driver_XXX():

    SESSION_THRESHOLD = 60

    user_code = np.array(df.user_code)
    click_time = np.array(df.time)

    res = np.empty(len (user_code), dtype = np.int)

    for u in range(user_code.min(), user_code.max() + 1):
        begin = np.searchsorted(user_code, u)
        end   = np.searchsorted(user_code, u+1)

        print(f"For user_code {u}: start index = {begin}, beyond end={end}")

        e = Close1D(click_time[begin:end], SESSION_THRESHOLD)

        if begin == 0:
            pass
        else:
            max_used = np.max(res[:begin])
            e = e + max_used +1 

        res[begin:end] = e

"""c"""    

import general.TimeLineTool as tl


def GetGroups(acData, anGroup):

    assert (anGroup.min() == 0)

    nGroups = anGroup.max() + 1

    l = []
    for u in range(0, nGroups):
        begin = np.searchsorted(anGroup, u)
        end   = np.searchsorted(anGroup, u+1)

        first_element = acData[begin]
        last_element = acData[end - 1]

        length = last_element - first_element + 1

        center = first_element + length/2

        element_count = end -1 - begin + 1

        density = element_count / length

        l.append( (center, length/2, element_count))

        # print(f"begin = {begin}, end = {end}, first = {first_element}, last = {last_element}, density = {density}")

    return l

"""c"""




def Analyze_Cluster(acData, nProximityValue):

    anGroup = tl.TimeLineTool_GetProximityGroups1D(acData, nProximityValue)

    l = GetGroups(acData, anGroup)

    score = 0

    for x in l:
        group_range = x[1] * 2
        element_count = x[2]
        density = element_count / group_range
        # print(f"range = {group_range}, element_count = {element_count}, density = {density} ")
        score = score + density * element_count

    return score
   
"""c"""

# Generate random events in range

df.ip_and_sys.value_counts()


m = df.ip_and_sys == 108753

q = df[m]

len (q)

acData = q.time.values
acData = np.sort(acData)

r = range(acData.max())

acRandomData = np.random.choice(r, len (acData))
acRandomData = np.sort(acRandomData)


global_density = len(acData) / (acData.max() - acData.min() + 1)
print(f"global_density {global_density}")

global_density_random = len(acRandomData) / (acRandomData.max() - acRandomData.min() + 1)
print(f"global_density_random {global_density_random}")

xRange = 4000
lcProx = range(xRange)

y_real = []
y_random = []
y_diff = []

for n, x_value in enumerate(lcProx):
    score_real = Analyze_Cluster(acData, x_value)
    score_rand = Analyze_Cluster(acRandomData, x_value)

    y_real.append(score_real)
    y_random.append(score_rand)
    y_diff.append(score_real - score_rand)


    print(f"{n/xRange}")

"""c"""

import matplotlib.pyplot as plt
plt.plot(lcProx, y_real)
plt.plot(lcProx, y_random)
plt.plot(lcProx, y_diff)
plt.show()


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







