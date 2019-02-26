
import numpy as np
import pandas as pd



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
        
        
        
    return ix_a, _a

"""c"""



df_1 = pd.DataFrame({'id': ix_a, 'a': _a})


df_1

df_1.id.max()




