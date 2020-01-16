


import numpy as np
import pandas as pd


df = pd.DataFrame({'id' : [0, 0, 1, 7, 2, 3, 4, 2, 2], 'v' : [3, 4, 1, 1, 4, 5, 9, 3, 4]})

df = df.sort_values(by = 'id').reset_index(drop = True)




def add_nth(df, g, idx):
    v_n = g.nth(idx)
    v_n.name = f'l_r{idx}'
    return df.join(v_n, on = ['id'])


g = df.groupby('id')['v']

for x in range(3):
    df = add_nth(df, g, x)