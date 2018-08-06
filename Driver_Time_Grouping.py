
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









