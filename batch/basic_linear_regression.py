
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :2]


def cost_function(X, m, b):

    sum = 0

    for x in X:
        y_true = x[1]
        y_pred = m * x[0] + b

        diff = y_true - y_pred
        diff2 = diff **2
        sum = sum + diff2
    
    return sum/len(X)

"""c"""

def grad_m_function(X, m, b):

    sum = 0

    for x in X:
        y_true = x[1]
        y_pred = m * x[0] + b
        
        diff = y_true - y_pred

        value = x[0] * diff

        sum = sum + value
            
    return -2 * sum/len(X)

"""c"""

def grad_b_function(X, m, b):

    sum = 0

    for x in X:
        y_true = x[1]
        y_pred = m * x[0] + b
        
        diff = y_true - y_pred

        sum = sum + diff
            
    return -2 * sum/len(X)

"""c"""

m = -11
b = 23

N = range(50000)

lr_m = 0.01
lr_b = 0.01

last_grad_m = grad_m_function(X, m, b)
last_grad_b = grad_b_function(X, m, b)

for i in N:
    C = cost_function(X, m, b)

    grad_m = grad_m_function(X, m, b)
    grad_b = grad_b_function(X, m, b)

    m = m - lr_m * grad_m
    b = b - lr_b * grad_b

    print(f"C= {C}, m= {m}, b= {b}, lr_m = {lr_m}, lr_b = {lr_b}, grad_m = {grad_m}, grad_b = {grad_b}")

    m_sign_const = False

    if grad_m > 0:
        m_sign_const = last_grad_m > 0
    else:
        m_sign_const = last_grad_m < 0

    if m_sign_const:
        pass
    else:
        lr_m = lr_m * 0.5


    b_sign_const = False
    
    if grad_b > 0:
        b_sign_const = last_grad_b > 0
    else:
        b_sign_const = last_grad_b < 0

    if b_sign_const:
        pass
    else:
        lr_b = lr_b * 0.5

    last_grad_m = grad_m
    last_grad_b = grad_b            


"""c"""




plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], color='b', label='0')
plt.show()



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