
# A study on ROC curves and AUC.
# Worker function and a driver further below.


import numpy as np
import pylab as pl



def roc_calc(y_t, y_p_rank, true_threshold):

    assert (len(y_t) == len (y_p_rank))

    nSamples = len (y_t)

    y_p_pos = y_p_rank[:true_threshold]

    y_p = np.zeros(nSamples, dtype=bool)

    y_p[y_p_pos] = True

    positive_samples = y_p.sum()
    negative_samples = len (y_p) - positive_samples

    print(f"    Positive samples: {positive_samples}")
    print(f"    Negative samples: {negative_samples}")

    y_tp = np.logical_and(y_p, y_t)

    true_positive = y_tp.sum()

    print(f"    True positives: {true_positive }")

    y_tn = np.logical_and(np.logical_not(y_p), np.logical_not(y_t))

    true_negative = y_tn.sum()

    print(f"    True negatives: {true_negative}")

    y_fp = np.logical_and(y_p, np.logical_not(y_t))

    false_positive = y_fp.sum()

    print(f"    False positives: {false_positive}")

    y_fn = np.logical_and(np.logical_not(y_p), y_t)

    false_negative = y_fn.sum()

    print(f"    False negative: {false_negative}")

    true_positive + true_negative + false_positive + false_negative

    true_positive_rate = true_positive/ ( true_positive + false_negative)

    print(f"    TPR: {true_positive_rate}")

    false_positive_rate = false_positive/ ( false_positive + true_negative)

    print(f"    FPR: {false_positive_rate}")

    return (false_positive_rate, true_positive_rate)

"""c"""

# 10 predictions. Ranked from 0 - most likely positive to 9, most likely negative:

y_p_rank = [1, 9, 2, 3, 0, 8, 6, 7, 4, 5]

y_t = [True, True, False, False, True, False, False, False, True, False]

nSamples = len (y_p_rank)

assert (nSamples == len (y_t))


true_threshold = 0

l_x = []
l_y = []


while (true_threshold <= nSamples):

    print(f"Threshold = {true_threshold}:")

    t = roc_calc(y_t, y_p_rank, true_threshold)

    print(f"    FPR = {t[0]}, TPR = {t[1]}")

    true_threshold = true_threshold + 1

    l_x.append(t[0])
    l_y.append(t[1])

"""c"""

l_nominal = [0,1]

pl.plot(l_nominal, l_nominal)

pl.plot(l_x,l_y)
pl.title('ROC')
pl.xlabel('FPR')
pl.ylabel('TPR')

pl.show()