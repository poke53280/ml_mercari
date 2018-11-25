

import numpy as np
import pandas as pd
import gc
from datetime import datetime

from timeit import default_timer as timer


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

    idx = np.arange(0,num_bins, num_bins/ num_elements)

    # Rasterize
    idx = idx.astype(int)
    
    y_cat_tot = np.empty(num_elements, dtype = int)

    y_cat_tot[y_arg_sort] = idx

    return y_cat_tot


####################################################################################
#
#   pimpim
#


def pimpim(df, offset, length, slot_size, num_bins_y):

    t = list()
    t.append(timer())
    

    x = df.mjd.values[offset: offset + length].copy()

    x = x - x.min()

    t.append(timer())

    p = df.passband.values[offset: offset + length]
    y_this_read_out = y_read_out[offset: offset + length]


    t.append(timer())

    num_bins_x = int (.5 + x.max()/slot_size)

    bins = np.linspace(0,  x.max(), num_bins_x, endpoint = False)

    digitized = np.digitize(x, bins)
    digitized = digitized - 1

    out = np.empty((6, num_bins_x), dtype = np.float32)
    out[:, :] = np.nan

    t.append(timer())

    for b in range (6):

        m = (p == b)

        y_p = y_this_read_out[np.where(m)]
        d_p = digitized[np.where(m)]

        out[b, d_p] = y_p

    t.append(timer())

        
    colsum = np.nansum(out, axis = 0)
    m = colsum == 0

    out[0, m] = num_bins_y * 6

    out[1, :] += (num_bins_y * 1)
    out[2, :] += (num_bins_y * 2)
    out[3, :] += (num_bins_y * 3)
    out[4, :] += (num_bins_y * 4)
    out[5, :] += (num_bins_y * 5)

    t.append(timer())
   


    # !!! No zeroes/Padding value.

    out = out.flatten(order = 'F')
    m = np.isnan(out)
    out = out[~m]

    t.append(timer())

    l, start, value = rle(out)

    m = (value == num_bins_y * 6)

    t.append(timer())

    l = l[m]
    start = start[m]
    out[start] += l

    m = (out == num_bins_y * 6)

    out = out[~m]

    t.append(timer())

    return out, t
"""c"""


def get_train_set(filename):
    store = pd.HDFStore(filename)

    # run length stop indices
    df_idx = pd.read_hdf(store, 'idx')

    idx = np.array(df_idx[0], dtype = np.int32)

    idx = np.insert(idx, 0, 0)

    begin_offset = idx[:-1]
    lengths = np.diff(idx)

    del df_idx
    del idx

    gc.collect()

    assert begin_offset.shape[0] == lengths.shape[0]

   
    df = store.select('data', start = begin_offset[-7848] , stop = begin_offset[-1] + lengths[-1] )
    df = df.reset_index(drop = True)

    return df, begin_offset, lengths

"""c"""


def get_first_items(nFirstItems, filename):
    store = pd.HDFStore(filename)

    # run length stop indices
    df_idx = pd.read_hdf(store, 'idx')

    idx = np.array(df_idx[0], dtype = np.int32)

    idx = np.insert(idx, 0, 0)

    begin_offset = idx[:-1]
    lengths = np.diff(idx)

    del df_idx
    del idx

    gc.collect()

    assert begin_offset.shape[0] == lengths.shape[0]

    print(f"num items = {begin_offset.shape[0]}")

    assert nFirstItems <= begin_offset.shape[0]

    iLastElement = nFirstItems - 1

    iBeyondLast = begin_offset[iLastElement] + lengths[iLastElement]
    
    df = store.select('data', start = 0 , stop = iBeyondLast )
    df = df.reset_index(drop = True)

    return df, begin_offset, lengths
"""c"""


# Generate snippets 

num_snip_per_row = 10

num_rows = anDataConst.shape[0]

num_snippets = num_snip_per_row * num_rows

sample_init_size = 20

anSnippet = np.zeros((num_snippets, sample_init_size), dtype = np.uint16)
aSnippetSize = np.empty(num_snippets, dtype = np.uint16)


iSnippet = 0

for iRow in range(num_rows):

    if iRow % 10000 == 0:
        print(f"Row {iRow} / {num_rows}")

    anDataRow = anDataConst[iRow].copy()

    iMin = 0
    iMax = value_area[iRow] - sample_init_size

    iFirstBreakChar = 6 * num_bins_y

    m = anDataRow >= iFirstBreakChar

    for i in range(num_snip_per_row):

        iOffset = np.random.choice(range(iMin, iMax +1 ))

        data_raw = anDataRow[iOffset : iOffset + sample_init_size]

        m_this = m[iOffset : iOffset + sample_init_size]

        iCut = sample_init_size

        if m_this.sum() > 0:
            iCut = np.where(m_this)[0][0]

        anSnippet[iSnippet, 0:iCut] = data_raw[0:iCut]
        aSnippetSize[iSnippet] = iCut

        iSnippet = iSnippet + 1

"""c"""

# Save snippets

np.save(DATA_DIR + "anSnippet_all", anSnippet)
np.save(DATA_DIR + "aSnippetSize_all", aSnippetSize)







DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE



df_meta_test = pd.read_csv(DATA_DIR + "test_set_metadata.csv")
num_total_test = np.unique(df_meta_test.object_id.values).shape[0]


df_meta = pd.read_csv(DATA_DIR + "training_set_metadata.csv")
num_total_training = np.unique(df_meta.object_id.values).shape[0]

num_objects = num_total_test + num_total_training


# num items = 3500738
df, begin_offset, anLength = get_first_items(num_objects, DATA_DIR + 'data.h5')


assert num_objects == np.unique(df.object_id.values).shape[0]


np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=200)
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



anData = np.zeros((num_objects,num_sequence_length), dtype = np.uint16)   

res_size = np.zeros(num_objects, dtype = np.uint16)

timers = np.zeros(100, dtype = np.float32)

for ix in range(num_objects):

    if ix % 100000 == 0:
        print(f"{ix} {num_objects}...")

    offset_this = begin_offset[ix]
    length = anLength[ix]

    res, t = pimpim(df, offset_this, length, slot_size, num_bins_y)

    t_diff = np.diff(np.array(t))

    timers[0:t_diff.shape[0]] += t_diff

    res_size[ix] = res.shape[0]

    res = res.astype(np.uint16)
    res = res[:num_sequence_length]
    anData[ix, 0:res.shape[0]] = res




# Values from 0 to value_area (excl) for all rows.
value_area = np.clip(res_size, None, num_sequence_length)


np.save(DATA_DIR + "anData_all", anData)
np.save(DATA_DIR + "value_area_all", value_area)


