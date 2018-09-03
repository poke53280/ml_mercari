
import numpy as np
import time

import pandas as pd

nRuns = 10

def sort_seq(d):

    N = d.shape[0]
    d = np.sort(d)

    return d

N = 900000000

d0 = np.empty(N, dtype = np.int32)
d1 = np.empty(N, dtype = np.int32)
d2 = np.empty(N, dtype = np.int32)

d_res = np.empty(N, dtype = np.int32)

r = range(10)

for x in r:

    start = time.time()

    d_res = d0 + d1 - d2

    end = time.time()

    dtime = end - start

    print(f"Processing time: {dtime:.1f} s.")

df = pd.DataFrame([d0, d1, d2, d_res])


with ThreadPool(processes=8) as pool:
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        y_pred = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)

