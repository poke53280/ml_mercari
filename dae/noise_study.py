

s = SwapNoise()

X0 = np.zeros((5, 10), dtype='float32')
X1 = np.zeros((5, 10), dtype='float32')

p = 0.15

nCols = X0.shape[1]
nRows = X0.shape[0]

nNoiseCols = int (nCols * p + .5)

anCols = range(nNoiseCols)

X1[:, anCols] = 1

for r in range(X1.shape[0]):
    np.random.shuffle(X1[r])


np.random.shuffle(X1)


X1

row = [1,0,0,0,0]

np.random.shuffle(X0[2])

X0[:, 3] = 11

X0


