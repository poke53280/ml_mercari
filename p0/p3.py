###############################################################################
#
# Analysis - timings.
#
#


import pandas as pd
import numpy as np
import more_itertools as mit


DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


########################################################################
#
#    check_period
#
#

def check_period(w):
    aF0 = np.array(w.F0)

    assert np.unique(aF0).shape[0] == 1

    f0 = aF0[0]

    aF1 = np.array(w.F1)     # Inclusive
    aT0 = np.array(w.T0) +1  # Exclusive

    min = aF1.min()
    max = aT0.max()

    period_size = max - min

    print(f"Analyzing period {f0} with {len(aF1)} interval(s) ...")
    
    assert period_size > 0

    acCount = np.zeros(shape = period_size)

    for (f, t) in zip (aF1, aT0):

        m = np.zeros(shape = period_size, dtype = np.bool)

        assert f-min >= 0
        assert t-min <= period_size

        m[f - min:t - min] = True

        acCount[m] = acCount[m] + 1

    acCountCats = np.unique(acCount).astype('int')

    for cat in acCountCats:
        m = (acCount == cat)

        rPct = 100.0 * m.sum() / period_size

        if cat != 1:
            print(f"   Warning: {cat} coverage for {rPct:.1f}% days")


    if f0 != min:
        print(f"    Warning: Period start f0 = {f0} unequal period start = {min}")

    print(f"   Interval [{min}, {max}], L = {max - min} ")

    return (min, max)
    

########################################################################
#
#    get_periods_rasterized
#

def get_periods_rasterized(df, idx):
    
    m = (df.ID == idx)
    w = df[m]

    aF1 = np.array(w.F1)     # Inclusive
    aT0 = np.array(w.T0) +1  # Exclusive

    min = aF1.min()
    max = aT0.max()

    period_size = max - min

    assert period_size > 0

    acCount = np.zeros(shape = period_size)

    for (f, t) in zip (aF1, aT0):

        m = np.zeros(shape = period_size, dtype = np.bool)

        assert f-min >= 0
        assert t-min <= period_size

        m[f - min:t - min] = True

        acCount[m] = acCount[m] + 1

    m = acCount >= 1

    acData = np.arange(min, max)

    acData = acData[m]

    return acData


########################################################################
#
#    check_periods_together
#
   
def check_periods_together(df, idx):
   
    acData = get_periods_rasterized(df, idx)

    l = [list(group) for group in mit.consecutive_groups(acData)]

    nGroups = len (l)

    print(f"Clustering to {nGroups} group(s)")

"""c"""


########################################################################
#
#    check_periods
#
#

def check_periods(df, idx):
    m = (df.ID == idx)

    q = df[m]

    aF0 = np.unique(np.array(q.F0))

    # print(f"Analyzing idx = {idx}. {len(aF0)} period(s) in total.")

    lMin = []
    lMax = []


    for f0 in aF0:
        m = (q.F0 == f0)
        w = q[m]

        (min, max) = check_period(w)

        lMin.append(min)
        lMax.append(max)


    # Analyse in period vs outside period

    acMin = np.array(lMin)
    acMax = np.array(lMax)

    c = np.empty((acMin.size + acMax.size,), dtype=np.int32)
    c[0::2] = acMin
    c[1::2] = acMax

    ediff = np.ediff1d(c)

    in_period = ediff[0::2]
    off_period = ediff[1::2]

    if len(off_period) > 0:
        off_min = off_period.min()
        print(f" Off min = {off_min}")

    else:
        print(f" Off min = NA")

"""c"""

df = pd.read_pickle(DATA_DIR + "noised_30JUL2018_cleaned.pkl")

df.shape[0]

df = df.drop_duplicates()

df.shape[0]



#
# 97 - to and from. Negative overlaps
# 98 - Negative overlaps
#

 
def desc_periods(df, idx):

    acData = get_periods_rasterized(df, idx)
    l = [list(group) for group in mit.consecutive_groups(acData)]

    l_min = []
    l_max = []

    for x in l:
        l_min.append(x[0])
        l_max.append(x[-1])


    acMin = np.array(l_min)
    acMax = np.array(l_max)

    acMax = acMax + 1   # Exclusive max


    c = np.empty((acMin.size + acMax.size,), dtype=np.int32)

    c[0::2] = acMin
    c[1::2] = acMax

    ediff = np.ediff1d(c)


    off_period = ediff[1::2]

    info = str(idx) + " "

    if len(off_period) > 0:
        min_off = np.min(off_period)

        if min_off < 4:
            info = info + f" Warning: minF[{min_off}] "

    toggle = 0

    for x in ediff:

        if toggle % 2 == 0:
            info = info + "S[" + str(x) + "] "

        else:
            info = info + "F[" + str(x) + "] "


        toggle = toggle + 1

    return info


N = 50000
warn_count = 0

for x in range(N):
    info = desc_periods(df, x)

    if "arning" in info:
        print(info)
        warn_count = warn_count + 1

"""c"""

rPct = 100.0 * warn_count / N

print(f"N = {N}, warnings = {warn_count}, {rPct:.1f}%")


