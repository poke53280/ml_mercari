
import pandas as pd
import numpy as np

def create_sortable(fid, reg):
    zFid = fid.map(lambda x: f'{x:011}')
    zReg = reg.astype(int).map(lambda x: f'{x:05}')
    zID = zFid + zReg
    id = pd.to_numeric(zID, errors = 'raise', downcast = 'unsigned')

    return id


l_fid0 = [6, 12345223344, 12345223344, 12345223344, 12345223344, 12345223344, 1345223344, 1345223344, 1345223344, 1345223344, 3]
l_reg0 = [1.5, 5.2, 9.2, 3.4, 7.5, 11.4, 1.1, 1.9, 2.0, 4.7, 4.9]
l_data0 = [11, 99, 21, 23, 0, 23, 31, 31, 32, 99, 11]


# XML - often slightly ahead
df0 = pd.DataFrame({'fid':l_fid0, 'reg': l_reg0, 'data0': l_data0})


l_fid1 = [12345223344, 12345223344, 12345223344, 12345223344, 12345223344, 1345223344, 1345223344, 1345223344, 1345223344, 3]
l_reg1 = [2.2, 3.1, 4.7, 4.7, 2.4, 7.1, 3.1, 9.3, 2.2, 3.2]
l_data1 = [90, 91, 12, 3, 10, 39, 22, 19, 10, 10]

# Tab - often slightly behind
df1 = pd.DataFrame({'fid':l_fid1, 'reg': l_reg1, 'data1': l_data1})




# Create sortable column in fid and reg time

df0 = df0.assign(id = create_sortable(df0.fid, df0.reg))
df1 = df1.assign(id = create_sortable(df1.fid, df1.reg))

df0 = df0.sort_values(by = 'id').reset_index(drop = True)
df1 = df1.sort_values(by = 'id').reset_index(drop = True)


# Store index
df0 = df0.reset_index()
df0 = df0.rename(columns = {'index': 'index0'})

df_m = pd.merge_asof(df1, df0, on = 'id', direction = 'backward')

# Fid_Y is now float if there were nans

nfid_y = df_m.fid_y.copy()

m_nan = nfid_y.isna()

nfid_y = nfid_y.fillna(0).astype(np.uint64) # no fid_x is 0

df_m = df_m.assign(fid_y = nfid_y)

m_fid_match = (df_m.fid_x == df_m.fid_y)

df_m = df_m.assign(m_fid_match = m_fid_match)


d_reg = df_m.reg_x - df_m.reg_y


df_m = df_m.assign(d_reg = d_reg)


# Nanify 

l_to_nan = ['data0', 'd_reg', 'reg_y', 'index0']

for x in l_to_nan:
    s = df_m[x].copy()

    s[~m_fid_match] = np.NaN
    df_m = df_m.assign(**{x: s})



df_m = df_m.drop(['fid_y', 'id', 'reg_y', 'm_fid_match'], axis = 1)

df_m = df_m.assign(index0 = df_m.index0.fillna(-1).astype(int))


data0 = df0.data0


data_forward = df_m.index0.map(lambda x: data0.iloc[x] if x >= 0 else -1)

df_m = df_m.assign(data_forward = data_forward)