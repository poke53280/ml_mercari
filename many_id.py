
import numpy as np
import pandas as pd
import networkx as nx

def add_single_elements(ix_a, _a, i_a0):

    empty = ix_a.shape[0] == 0


    # Get the unique IDs

    i_a0_unique = np.unique(i_a0)


    m = np.in1d(i_a0_unique, _a)


    # ~m element(s) are not present. Add them to array with unique unused idx

    
    if empty:
        id_0 = 0
    else:
        id_0 = np.max(ix_a) +1 

    id_new = np.arange(start = id_0, stop = id_0 + (~m).sum(), dtype= np.uint64)

    ix_a = np.concatenate((ix_a, id_new))

    _a = np.concatenate((_a, i_a0_unique[~m]))

    assert np.unique(_a).shape == _a.shape

    return ix_a, _a

"""c"""



ix_a = np.array([], dtype = np.uint64)
_a = np.array([], dtype = np.uint64)


assert np.unique(_a).shape == _a.shape

# Add some extras:

i_ex0 = np.array([9, 3, 100005], dtype = np.uint64)
ix_a, _a = add_single_elements(ix_a, _a, i_ex0)


#i_a0 = np.arange(0, 9, dtype = np.uint64)
#i_a1 = np.arange(8, 8 + 9, dtype = np.uint64)

i_a0 = np.random.randint(0, 50 * 1000 * 1000, 3 * 1000 * 1000)
i_a1 = np.random.randint(0, 50 * 1000 * 1000, 3 * 1000 * 1000)


def add_pairs(ix_a, _a, i_a0, i_a1):

    assert i_a0.shape == i_a1.shape

    i_a = np.concatenate((i_a0, i_a1))

    # Get uniqueness mask
    u, c = np.unique(i_a, return_counts=True)

    idx = np.searchsorted(u, i_a)

    # Unique element at m True:
    m_unique = c[idx] == 1

    m_existing = np.in1d(i_a, _a)

    m_new_and_unique = m_unique & ~m_existing

    # Split and check pairs
    m_pair = np.split(m_new_and_unique, 2)

    # Both numbers in these pairs are new and unique
    m_combines = m_pair[0] & m_pair[1]


    ix_a, _a = add_single_elements(ix_a, _a, i_a0[~m_combines])
    ix_a, _a = add_single_elements(ix_a, _a, i_a1[~m_combines])


    idx_sorted = np.argsort(_a)

    a_sorted = _a[idx_sorted]
    g_sorted = ix_a[idx_sorted]

    idx_a0 = np.searchsorted(a_sorted, i_a0[~m_combines], side='left', sorter=None)
    idx_a1 = np.searchsorted(a_sorted, i_a1[~m_combines], side='left', sorter=None)


    g0 = g_sorted[idx_a0]
    g1 = g_sorted[idx_a1]

    # Replace to smaller group numbers

    m_min = g0 < g1

    g_min = g0.copy()

    g_min[~m_min] = g1[~m_min]

    g_max = g1.copy()

    g_max[~m_min] = g0[~m_min]

    idx = np.lexsort([g_min, g_max])[::-1]

    g_min = g_min[idx]
    g_max = g_max[idx]

    from_res = g_max.copy()
    to_res = g_min.copy()




    # Todo: (Possible). g_max adjustment - sorted can scan and replace
    # Todo: Can overwrite starting from index, no need to rewrite full array
    # Todo: Possibly use index array for look ups. (Memory is not a problem).

    for i in range(g_min.shape[0]):


        if i % 1000 == 0:
            print(f"{i +1} of {g_min.shape[0]}")

        from_num = g_max[i]
        to_num = g_min[i]

        from_res[i] = from_num
        to_res[i] = to_num
    
        m = g_max == from_num
        g_max[m] = to_num
    
        m = g_min == from_num
        g_min[m] = to_num

    """c"""


    m = from_res == to_res

    from_res = from_res[~m]
    to_res = to_res[~m]

    for i in range(from_res.shape[0]):

        if i % 1000 == 0:
            print(f"{i +1} of {from_res.shape[0]}")

        from_g = from_res[i]
        to_g = to_res[i]

        # print(from_g, to_g)

        m = ix_a == from_g
        ix_a[m] = to_g
    
    """c"""

    # New groups can simply be formed by assigning the same unique group number to both

    ix_a, _a = add_single_elements(ix_a, _a, i_a0[m_combines])
    ix_a, _a = add_single_elements(ix_a, _a, i_a1[m_combines])


    num_basic = m_combines.sum()

    ix_a[-num_basic:] = ix_a[- 2 * num_basic: -num_basic]
        
    _a.shape
    ix_a.shape
    # Num elements: 5654509

    np.unique(ix_a).shape
    # Num groups: 2654509
        
    return ix_a, _a

"""c"""

a = np.column_stack((i_a0, i_a1))

G = nx.Graph()

G.add_edges_from(a)
G.add_nodes_from(i_ex0)

a = nx.to_pandas_dataframe(G)
# Number of nodes: 5654509




nx.number_connected_components(G)
# 2654509



# Fast:

l_out0 = []

iCount = 100000

for x in nx.connected_components(G):
    l_out0.append(x)
    iCount = iCount - 1
    if iCount == 0:
        break
"""c"""

# Slow 12.07 - stop after 8 minutes.
sg = list(nx.connected_component_subgraphs(G))

#d = nx.to_dict_of_lists(G)
#nodes on keys, and some neighbours but not all on values (?)

###########################################################################

import numpy as np

mm = MultiIDMap()

iTypeFID = 0
iTypeAID = 1

# Map a set of FIDs to a set of AIDs
anFID = np.random.randint(0, 100, 10, dtype =np.uint64)
anAID = np.random.randint(0, 100, 10, dtype =np.uint64)


anFID_id = mm.convert_raw(anFID, iTypeFID)
anAID_id = mm.convert_raw(anAID, iTypeAID)

mm.get_raw_id(anFID_id)


id = mm.convert_raw(np.array([8, 100, 72, 52], dtype =np.uint64), 0)

mm.get_raw_id(id)
mm.get_raw_type(id)

# Map a set of FIDs to other FIDs



class MultiIDMap:
    
    _raw_id = np.array([], dtype = np.uint64)
    _raw_t = np.array([], dtype = np.uint8)
    
    def __init__(self):
        pass

    def add(self, id_in, id_type):
        assert id_type <= np.iinfo(np.uint8).max
        assert id_in.dtype == np.uint64

        id_in_unique = np.unique(id_in)

        # The numbers from distribution that we have already got:
        m_t1 = self._raw_t == id_type

        m = np.in1d(id_in_unique, self._raw_id[m_t1])

        # Numbers to add as id_type:
        self._raw_id = np.concatenate((self._raw_id, id_in_unique[~m]))

        # Add to type array
        self._raw_t = np.concatenate((self._raw_t, np.zeros(id_in_unique[~m].shape, dtype = np.uint8 ) + id_type))

    def get_id(self, id_in, id_type):

        m_t1 = self._raw_t == id_type

        i_a_single_type = self._raw_id[m_t1]
        i_a_idx = np.arange(self._raw_id.shape[0])[m_t1]

        idx = np.argsort(i_a_single_type)

        i_a_single_type = i_a_single_type[idx]
        i_a_idx = i_a_idx[idx]

        assert np.in1d(id_in, i_a_single_type).all()

        find_all = np.searchsorted(i_a_single_type, id_in)

        res_idx = i_a_idx[find_all]

        assert (self._raw_id[res_idx] == id_in).all(), "Missing elements"

        return res_idx

    def convert_raw(self, id_in, id_type):
        self.add(id_in, id_type)
        return self.get_id(id_in, id_type)

    def get_raw_id(self, id):
        return self._raw_id[id]

    def get_raw_type(self, id):
        return self._raw_t[id]

"""c"""



