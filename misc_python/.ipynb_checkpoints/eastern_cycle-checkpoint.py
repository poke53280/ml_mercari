

import numpy as np

an = np.array([100, 300, 400, 600, 900])

l_a_mid = []
l_L = []

for idx in range(1, an.shape[0] - 1):

    a_low = an[idx - 1]
    a_mid = an[idx]
    a_hi  = an[idx + 1]

    L = a_mid + 0.5 * a_hi - 0.5 * a_low

    l_a_mid.append(a_mid)
    l_L.append(L)

an_mid = np.array(l_a_mid)
an_L = np.array(l_L)


an_day = np.array(range(101, 900))

m = (an_day > an.min()) & (an_day < an.max())

assert m.all()


l_cos = []

for x in an_day:
    
    idx = np.argsort(np.abs(an_mid - x))[0]

    a_mid = an_mid[idx]
    L = an_L[idx]

    l_cos.append(np.cos(2*np.pi*(x-a_mid)/ L))






