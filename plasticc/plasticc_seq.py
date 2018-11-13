

import numpy as np
import pandas as pd
import gc
from datetime import datetime


def rle(inarray):
        """ returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                  # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

####################################################################################
#
#   get_binned_sequence
#
#

def get_binned_sequence(y, num_bins):

    num_elements = y.shape[0]

    y_s = np.sort(y)
    y_arg_sort = np.argsort(y)

    elements_per_bin = int (.5 + num_elements/num_bins)

    idx_lo = []
    idx_hi = []

    for b in range(num_bins):

        idx_lo.append(b  * elements_per_bin)

        if b == num_bins - 1:
            idx_hi.append(num_elements)
        else:
            idx_hi.append( (b + 1) * elements_per_bin)


    # Test bin count:

    y_c = np.empty(y_s.shape, dtype = np.int32)

    for i, (a, b) in enumerate (zip (idx_lo, idx_hi)):

        nElements = b - a

        y_c[a:b] = i

        print(f"{i}: {nElements}")


    y_cat_tot = np.empty(y_c.shape, dtype = np.int32)
    y_cat_tot[:] = np.nan

    y_cat_tot[y_arg_sort] = y_c

    assert np.isnan(y_cat_tot).sum() == 0
    assert np.min(y_cat_tot) == 0
    assert np.max(y_cat_tot) == (num_bins - 1)

    return y_cat_tot




DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

df_meta = pd.read_csv(DATA_DIR + "training_set_metadata.csv")
df = pd.read_csv(DATA_DIR + "training_set.csv")

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 480


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)


slot_size = 5


num_bins = 150


num_sequence_length = 200

y = df.flux.values
p = df.passband.values

### BEGIN binning experiments


y_err = df.flux_err.values

assert y.shape == y_err.shape


y_cat = get_binned_sequence(y, num_bins)

y_err_cat = get_binned_sequence(y_err, num_bins)


df = df.assign(y_cat = y_cat)

df = df.assign(y_err_cat = y_err_cat)

y_combo = y_cat * 150 + y_err_cat


np.unique(y_combo).shape


df = df.assign(y_g = y_combo)


s = df.y_g.astype('category')


q = s.value_counts()

an = q.values

m = an < 10



m = df.y_g == 22496

df[m]


### END binning experiments








y_read_out = np.empty(y.shape, dtype = np.int32)

for b in range(6):
    m = (p == b)
    y_read_out[m] = ((num_levels -1) * (y[m] - np.min(y[m])) / y[m].max()).astype(dtype = np.int32)


"""c"""


uniqueID = np.unique(df.object_id.values)

num_objects = uniqueID.shape[0]


start = datetime.now()

anData = np.zeros((num_objects,num_sequence_length), dtype = np.uint16)


res_size = []

for ix, x in enumerate(uniqueID):
    res = pimpim(df, x)

    res_size.append(res.shape[0])

    res = res.astype(np.uint16)
    res = res[:num_sequence_length]
    anData[ix, 0:res.shape[0]] = res


end = datetime.now()

dT = end - start
dSeconds = dT.total_seconds()

print(f"time: {dSeconds:.2f}s")




def pimpim(df, id):

    m = df.object_id == id

    x = df.mjd.values[np.where(m)]

    x = x - x.min()

    p = df.passband.values[np.where(m)]

    y_this_read_out = y_read_out[np.where(m)]

    num_bins = int (.5 + x.max()/slot_size)

    bins = np.linspace(0,  x.max(), num_bins, endpoint = False)

    digitized = np.digitize(x, bins)
    digitized = digitized - 1

    out = np.empty((6, num_bins), dtype = np.float32)
    out[:, :] = np.nan

    for b in range (6):

        m = (p == b)

        y_p = y_this_read_out[np.where(m)]
        d_p = digitized[np.where(m)]

        out[b, d_p] = y_p


    colsum = np.nansum(out, axis = 0)
    m = colsum == 0

    out[0, m] = num_levels * 6

    out[1, :] += (num_levels * 1)
    out[2, :] += (num_levels * 2)
    out[3, :] += (num_levels * 3)
    out[4, :] += (num_levels * 4)
    out[5, :] += (num_levels * 5)


    # !!! No zeroes/Padding value.

    out = out.flatten(order = 'F')
    m = np.isnan(out)
    out = out[~m]

    l, start, value = rle(out)

    m = (value == num_levels * 6)

    l = l[m]
    start = start[m]
    out[start] += l

    m = (out == num_levels * 6)

    out = out[~m]

    return out
"""c"""


# Replace parts


an = anData[3090,:]


# Replace mask. Leave all breaks, and don't introduce breaks.

# Leave all zeros, and don't introduce any zeros.

length = 20

m0 = an == 0

mb = an >= (num_levels * 6)


m_leave = m0 | mb


anReplace = range(4, 23)

n = m_leave.shape[0]

aMask = np.zeros(n)

aMask[anReplace] = 1

aMask = aMask.astype(np.bool)

mFinal = aMask & ~m_leave



Anoise = A


# Start and length






