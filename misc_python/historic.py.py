
import pandas as pd
import numpy as np

l_id = [0, 1, 2, 3, 4, 5]
l_b = [1 , 1, 5, 3, 2, 0]
l_l = [3, 4, 2, 4, 5, 4]
l_d = ['a', 'a', 'b', 'c', 'a', 'b']

t_cut = 2


df = pd.DataFrame({'id': l_id, 'b' : l_b, 'L': l_l, 'd': l_d})


df = df.assign(compl = df.b + df.L)

df = df.sort_values(by = 'compl')

df = df.assign(t_cut = df.b + t_cut)


