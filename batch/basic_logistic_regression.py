
#
# https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
#

# model: 
#
# h = sigmoid ( sum (theta*xi) + b)
#
# c(theta) = sum(-y_true * np.log(h) - (1 - y_true) * np.log(1 - h)) / N
#
# del c/ del theta




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :2]
y = (iris.target != 0) * 1

plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend();

plt.show()

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])

### !!! 
        for i in range(self.num_iter):
            
            # Run model at current theta set.
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)

            # Calculate gradient with respect to all thetas (vector operation)
            #gradient = np.dot(X.T, (h - y)) / y.size

            n = 50

            idx = np.random.choice(ar, n, replace = False)

            diff_batch = h[idx] - y[idx]

            xt_batch = X.T[:, idx]

            gradient = np.dot(xt_batch, diff_batch) / n
            
            # Reduce cost by stepping downhill in all dimensions
            
            self.theta -= self.lr * gradient
                
            if(self.verbose ==True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                loss = self.__loss(h, y)
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()

"""c"""

# Test interactive:

theta = np.zeros(X.shape[1])

# Run model forward for the full train set:
z = np.dot(X, theta)
h = 1 / (1 + np.exp(-z))

# Calculate gradient on full train set
#gradient = np.dot(X.T, (h - y)) / y.size


model = LogisticRegression(lr=0.1, num_iter=300000, verbose=True)

model.fit(X,y)


preds = model.predict(X)

(preds == y).mean()


model.theta

plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');


plt.show()