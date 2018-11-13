

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

    y_c = np.empty(y_s.shape, dtype = np.uint16)

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



####################################################################################
#
#   pimpim
#
#


def pimpim(df, id, slot_size, num_bins_y):

    m = df.object_id == id

    x = df.mjd.values[np.where(m)]

    x = x - x.min()

    p = df.passband.values[np.where(m)]

    y_this_read_out = y_read_out[np.where(m)]

    num_bins_x = int (.5 + x.max()/slot_size)

    bins = np.linspace(0,  x.max(), num_bins_x, endpoint = False)

    digitized = np.digitize(x, bins)
    digitized = digitized - 1

    out = np.empty((6, num_bins_x), dtype = np.float32)
    out[:, :] = np.nan

    for b in range (6):

        m = (p == b)

        y_p = y_this_read_out[np.where(m)]
        d_p = digitized[np.where(m)]

        out[b, d_p] = y_p


    colsum = np.nansum(out, axis = 0)
    m = colsum == 0

    out[0, m] = num_bins_y * 6

    out[1, :] += (num_bins_y * 1)
    out[2, :] += (num_bins_y * 2)
    out[3, :] += (num_bins_y * 3)
    out[4, :] += (num_bins_y * 4)
    out[5, :] += (num_bins_y * 5)


    # !!! No zeroes/Padding value.

    out = out.flatten(order = 'F')
    m = np.isnan(out)
    out = out[~m]

    l, start, value = rle(out)

    m = (value == num_bins_y * 6)

    l = l[m]
    start = start[m]
    out[start] += l

    m = (out == num_bins_y * 6)

    out = out[~m]

    return out
"""c"""



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


num_bins_y = 150

num_sequence_length = 200

y = df.flux.values
p = df.passband.values




y_read_out = np.empty(y.shape, dtype = np.uint16)

for b in range(6):
    m = (p == b)
    y_read_out[m] = get_binned_sequence(y[m], num_bins_y)

"""c"""


uniqueID = np.unique(df.object_id.values)

num_objects = uniqueID.shape[0]


start = datetime.now()

anData = np.zeros((num_objects,num_sequence_length), dtype = np.uint16)   # To empty after debug

res_size = np.zeros(num_objects, dtype = np.uint16)

for ix, x in enumerate(uniqueID):
    res = pimpim(df, x, slot_size, num_bins_y)

    res_size[ix] = res.shape[0]

    res = res.astype(np.uint16)
    res = res[:num_sequence_length]
    anData[ix, 0:res.shape[0]] = res


end = datetime.now()

dT = end - start
dSeconds = dT.total_seconds()

print(f"time: {dSeconds:.2f}s")

value_area = np.clip(res_size, 0, num_sequence_length)

# Values from 0 to value_area (excl) for all rows.

sample_init_size = 10

# Collect data snippets.



iRow = 0
iMin = 0
iMax = value_area[iRow] - sample_init_size

iOffset = np.random.choice(range(iMin, iMax))


data_raw = anData[iRow, iOffset: iOffset + sample_init_size]

m = data_raw >= 6 * num_bins_y

iCut = sample_init_size

if m.sum() > 0:
    iCut = np.where(m)[0][0]
   
if iCut > 0:
    data_out = data_raw[0:iCut]










# Replace parts
#
# Source:
# * Data interval fixed size from non null area of row.
# * Remove all breaks
# * Take note of source size
#
# * Destination: Offset into row inside the non null area so that source size will fit.
# * Keep all breaks

# Possibly get from other source for each break.





nMaxBreakInclusive = np.max(anData)
nMinBreakInclusive = 6 * num_bins_y

anData.shape












an = anData[3090,:]


# Leave all breaks, and don't introduce breaks.

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






