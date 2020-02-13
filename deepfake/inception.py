

#
# Based on: https://www.kaggle.com/unkownhihi/starter-kernel-with-cnn-model-ll-lb-0-69235 by 
#


# Code for generating dataset:
import pandas as pd
import numpy as np

id_itrgd = [0,1, 2, 3]
reg_itrgd = [4, 5, 8, 9]
data_itrgd = ['b', 'q', 'f', 'h']


id_eia = [0, 0, 0, 1, 3, 3]
reg_eia = [4, 3, 4, 7, 10, 11]
data_eia = [1.1, 1.3, 0.9, 1.7, 0.2, 1.1]

df_i = pd.DataFrame({'it_id':id_itrgd, 'it_reg': reg_itrgd, 'it_data': data_itrgd})
df_e = pd.DataFrame({'eia_id':id_eia, 'eia_reg': reg_eia, 'eia_data': data_eia})

# Create sortable string from number

z_id = df_i.it_id.map(lambda x: f"{x:03}")
z_reg = df_i.it_reg.map(lambda x: f"{x:03}")
z_v = "id_" + z_id + "_reg_" + z_reg

df_i = df_i.assign(z_v_i = z_v)

z_id = df_e.eia_id.map(lambda x: f"{x:03}")
z_reg = df_e.eia_reg.map(lambda x: f"{x:03}")
z_v = "id_" + z_id + "_reg_" + z_reg

df_e = df_e.assign(z_v_e = z_v)

df_e = df_e.sort_values(by = 'z_v_e').reset_index(drop = True)
df_i = df_i.sort_values(by = 'z_v_i').reset_index(drop = True)

s_z_all = pd.concat([df_e.z_v_e, df_i.z_v_i])

e_id = np.searchsorted(np.unique(s_z_all), df_e.z_v_e)
i_id = np.searchsorted(np.unique(s_z_all), df_i.z_v_i)

df_e = df_e.assign(v = e_id)
df_i = df_i.assign(v = i_id)


pd.merge_asof(df_e, df_i, on = 'v')

# et.c.

