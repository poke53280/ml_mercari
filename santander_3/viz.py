

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

from santander_3.RowStatCollector import get_aggs

# viz

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train_CONST = pd.read_csv(DATA_DIR + 'train.csv')


train_CONST = train_CONST.drop(['ID', 'target'], axis = 1)



test_CONST = pd.read_csv(DATA_DIR + 'test.csv')
test_CONST = test_CONST.drop(['ID'], axis = 1)


Xtr = csr_matrix(train_CONST)
Xte = csr_matrix(test_CONST)

from scipy.sparse import vstack

X = vstack([Xtr, Xte])


col_prop = {}

for col in range(X.shape[1]):
    an = np.array (X[:, col].todense())
    an = an.squeeze()    

    m = (an != 0)

    nonzero_list = an[m]

    v = np.array(nonzero_list, dtype = np.float32)


    data = get_aggs_lite(v, '')
    
    col_prop[col] = data

"""c"""


df = pd.DataFrame(data = col_prop).transpose()



df['kurtosis'] = df['kurtosis'].replace([np.inf, -np.inf], np.nan)
df['kurtosis'] = df['kurtosis'].fillna(0)


# Normalize

from sklearn.preprocessing import StandardScaler

s = StandardScaler()


for col in df.columns:
    df[col] = s.fit_transform(df[col].values.reshape(-1,1))


from sklearn.decomposition import TruncatedSVD

pca = TruncatedSVD(n_components=2)
pca_result = pca.fit_transform(df.values)

import matplotlib.pyplot as plt

plt.scatter(pca_result[:,0],pca_result[:,1],color='blue')


plt.show()

from sklearn.manifold import TSNE

n_sne = 7000

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.values)

print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


plt.scatter(tsne_results[:,0],tsne_results[:,1],color='red')
plt.show()

def distance_to(col_prop, iTestIdx):

    d_list = []
    
    data0 = col_prop[iTestIdx]
    
    rec_min = 100000000
    best_idx = -1

    for key, value in col_prop.items():

        if key == iTestIdx:
            print("Skipping self")
            d_list.append(0.0)
            continue

        d2 = (data0['sum'] - value['sum'])**2 + (data0['max'] - value['max'])**2 + (data0['count'] - value['count'])**2
        d_list.append(d2)

        if d2 < rec_min:
            rec_min = d2
            best_idx = key
            print(f"best idx = {key}, d2 = {rec_min}")

        #print(key, d2)

    print(iTestIdx, col_prop[iTestIdx])
    print(best_idx, col_prop[best_idx])

    return d_list

"""c"""

c = {}

c[4] = col_prop[4]
c[5] = col_prop[5]

data0 = c[5]

q = distance_to (col_prop, 299)

