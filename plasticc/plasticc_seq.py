

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

anDataConst = anData.copy()



value_area = np.clip(res_size, 0, num_sequence_length)

# Values from 0 to value_area (excl) for all rows.

sample_init_size = 10

# Collect data snippets.

num_snippets = 3000

anSnippet = np.zeros((num_snippets, sample_init_size), dtype = np.uint16)
aSnippetSize = np.empty(num_snippets, dtype = np.uint16)

for iSnippet in range(num_snippets):

    iRow = np.random.choice(range(0, anDataConst.shape[0]-1))
    iMin = 0
    iMax = value_area[iRow] - sample_init_size

    iOffset = np.random.choice(range(iMin, iMax))

    data_raw = anDataConst[iRow, iOffset: iOffset + sample_init_size]

    m = data_raw >= 6 * num_bins_y

    iCut = sample_init_size

    if m.sum() > 0:
        iCut = np.where(m)[0][0]

    anSnippet[iSnippet, 0:iCut] = data_raw[0:iCut]
    aSnippetSize[iSnippet] = iCut

"""c"""


# Post process - remove small runs

m = aSnippetSize < 2

anSnippet = anSnippet[~m]
aSnippetSize = aSnippetSize[~m]


anSnippet.shape

# Got snippet bank

# Apply to data

anData = anDataConst.copy()

num_shuffles = 10

for iShuffle in range(num_shuffles):

    for iRow in range(num_objects):

        iMin = 0
        iMax = value_area[iRow] - sample_init_size

        iOffset = np.random.choice(range(iMin, iMax))

        data_raw = anDataConst[iRow, iOffset: iOffset + sample_init_size]

        m = data_raw >= 6 * num_bins_y

        iCut = sample_init_size

        if m.sum() > 0:
            iCut = np.where(m)[0][0]

        # Can replace values 0..iCut

        r = list (range(0, anSnippet.shape[0]))

        iSnippetRow = np.random.choice(r)

        anSnippetData = anSnippet[iSnippetRow]
        nSnippetSize = aSnippetSize[iSnippetRow]

        nReplaceRun = np.amin([nSnippetSize, iCut])

        anData[iRow, iOffset: iOffset + nReplaceRun] = anSnippetData[0:nReplaceRun]


anData.shape
anDataConst.shape

m = anData == anDataConst

nEqual = m.sum()
nAll = anData.shape[0] * anData.shape[1]

nDiff = nAll - nEqual
rDiff = 100.0 * nDiff / nAll

print(f"Diff: {rDiff:.1f}%")