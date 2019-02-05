
import numpy as np
import pandas as pd


##########################################################################
#
#       get_both_unique
#
#
#    Value in column a must be a unique value for that column.
#    Valie in column b must be a unique value for that column.
#

def get_both_unique(a, b):

    assert a.shape[0] == b.shape[0]
   
    a_idx = np.argsort(a)
    b_idx = np.argsort(b)

    b_idx_back = np.argsort(b_idx)
    a_idx_back = np.argsort(a_idx)

    au, an = np.unique(a, return_counts = True)
    bu, bn = np.unique(b, return_counts = True)

    m_a = an == 1
    m_b = bn == 1

    a = a[a_idx]
    b = b[a_idx]
    m_a_is_unique = np.in1d(a, au[m_a])

    a = a[a_idx_back]
    b = b[a_idx_back]
    m_a_is_unique = m_a_is_unique[a_idx_back]

    a = a[b_idx]
    b = b[b_idx]
    m_a_is_unique = m_a_is_unique[b_idx]
    m_b_is_unique = np.in1d(b, bu[m_b])

    m_unique = m_a_is_unique & m_b_is_unique

    a = a[b_idx_back]
    b = b[b_idx_back]
    m_unique = m_unique[b_idx_back]

    return m_unique
"""c"""

def example():

    a = np.random.choice(2410000, 4000000)
    b = np.random.choice(1420000, 4000000)

    df = pd.DataFrame(data = {'a': a, 'b': b})

    m = get_both_unique(np.array(df.a), np.array(df.b))

    df = df.assign(unique = m)

    df.shape
    df[m].shape
"""c"""
