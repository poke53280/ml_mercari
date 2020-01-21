

import numpy as np
import pandas as pd

from sklearn.manifold import MDS
from sklearn.manifold import smacof

import matplotlib.pyplot as plt
import seaborn as sns



sns.set()

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 1500)


towns = ['gjøvik', 'skien', 'oslo', 'kristiansand', 'lillehammer']

l_d = [('gjøvik', 'skien', 187), ('gjøvik', 'kristiansand', 332),
        ('skien', 'kristiansand', 150), ('gjøvik', 'lillehammer', 38), ('kristiansand', 'lillehammer', 358),
       ('oslo', 'skien', 102), ('lillehammer', 'skien', 218)]



, ('gjøvik', 'oslo', 98),  ('oslo', 'lillehammer', 135)]

# Validate towns:

for x in l_d:
    assert x[0] in towns, f"Unknown town: {x[0]}"
    assert x[1] in towns, f"Unknown town: {x[1]}"


num_cities = len(towns)
d_city = {}

for idx, c in enumerate (towns):
    d_city[c] = idx

X = np.empty((num_cities, num_cities), dtype = np.float64)

X[:] = 0

np.fill_diagonal(X, 0)



for x in l_d:
    iCity0 = d_city[x[0]]
    iCity1 = d_city[x[1]]
    dist = x[2]

    print (iCity0, iCity1, dist)

    X[iCity0, iCity1] = dist
    X[iCity1, iCity0] = dist


res = smacof(X, n_components=2, random_state=1, metric = True, verbose = 1, n_init = 10, eps=1e-12, max_iter=3000)


cities = towns
coords = res

# dist(df, 'skien', 'oslo', d_city)


df = pd.DataFrame({
    'x': coords[0][:, 0],
    'y': coords[0][:, 1],
    'group': cities
})

def dist (df, t0, t1, d_city):

    a = d_city[t0]
    b = d_city[t1]


    dx = df.iloc[a].x -  df.iloc[b].x
    dy = df.iloc[a].y -  df.iloc[b].y
    d2 = dx * dx + dy * dy
    return np.sqrt(d2)

# Todo: Maintain scale

p1=sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", color="red", scatter_kws={'s':100, 'alpha':0.7})
 
# add annotations one by one with a loop
for line in range(0,df.shape[0]):
     p1.text(df.iloc[line].x, df.iloc[line].y, df.iloc[line].group, horizontalalignment='left', size='medium', color='darkblue', weight='light')
 
plt.gca().set_aspect('equal', adjustable='box')

plt.show()

