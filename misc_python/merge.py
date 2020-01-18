
import pandas as pd
import numpy as np

MAX_DIAGNOSIS_AGE = 5

l_id = [0, 2, 3, 4, 4, 9]
l_reg = [1 , 1, 5, 3, 2, 0]

df = pd.DataFrame({'id': l_id, 'reg' : l_reg})

l_id = [0, 2, 5, 4, 9]
l_reg = [3, 2, 4, 2, 11]
l_value = [9, 134, 12, 2, 17]

df_x = pd.DataFrame({'id': l_id, 'reg': l_reg, 'd' : l_value})

reg_max = np.max([df_x.reg.max(), df.reg.max()])

reg_offset = reg_max + 1

df_x = df_x.assign(idx = df_x.reg + reg_offset * df_x.id)

assert (df_x.reg == df_x.idx%reg_offset).all()
assert (df_x.id == df_x.idx//reg_offset).all()

df = df.assign(idx = df.reg + reg_offset * df.id)

assert (df.reg == df.idx%reg_offset).all()
assert (df.id == df.idx//reg_offset).all()

df = df.sort_values(by = 'idx').reset_index(drop = True)
df_x = df_x.sort_values(by = 'idx').reset_index(drop = True)

idx_x = np.array(df_x.idx)
idx_t = np.array(df.idx)

i_hit = np.searchsorted(idx_x, idx_t, side = 'right')

i_hit = i_hit - 1

m_exists = (i_hit >= 0) & (i_hit < df_x.shape[0])

i_hit[~m_exists] = -1

x_id = np.empty_like(np.array(df.id))
x_id[:] = -2

x_id[~m_exists] = -1
x_id[m_exists] = df_x.loc[i_hit[m_exists]].id

assert (x_id != -2).all()

id_match = (df.id >=0) & (df.id == x_id)

i_hit[~id_match] = -1

df = df.assign(hit = i_hit)

df_x = df_x.assign(index = df_x.index)

df = df.merge(df_x[['d', 'index', 'reg', 'id']], how = 'left', left_on = df.hit, right_on = 'index')

del df_x

df = df.drop(['index', 'hit'], axis = 1)

d = df.d.fillna(0).astype(np.int32)
df = df.drop('d', axis = 1)

reg_diag = df.reg_y.fillna(0).astype(np.int32)
df = df.drop('reg_y', axis = 1)

diagnose_age = df.reg_x - reg_diag

m = diagnose_age >= 0

d[~m] = 0

m = diagnose_age < MAX_DIAGNOSIS_AGE

d[~m] = 0

df = df.assign(d = d)

df = df.drop(['idx'], axis = 1)

df = df.assign(id_y = df.id_y.fillna(0).astype(np.int32))

assert (df[df.d !=0].id_x == df[df.d !=0].id_y).all()

df = df.drop(['id_y'], axis = 1)


npuint64_info = np.iinfo(np.uint64)

npuint64_info.max





