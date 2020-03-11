
import numpy as np
import matplotlib.pyplot as plt




import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from DataLoader import load_clives
from DataLoader import get_clive_files


def error_fit_to_poly(x, y, degree):
    polynomial_features= PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)

    return (rmse, r2)







l_files = get_clive_files()


df = load_clives([l_files[0]])

anData = np.stack(df.test.values)


anData0 = anData[:, :, 0]

x_ = np.arange(anData0.shape[1])

x = x_[:, np.newaxis]

l_x_poly = []

model = LinearRegression()

for i in range(50):
    polynomial_features= PolynomialFeatures(degree=i)
    x_poly = polynomial_features.fit_transform(x)
    l_x_poly.append(x_poly)



l_rms = []
l_p = []

for y_ in anData0[:1000]:

    # transforming the data to include another axis
   
    y = y_[:, np.newaxis]


    l_rms = []
    for degree in [0, 1, 2, 3, 4, 10, 11, 12, 13, 20]:
        x_poly = l_x_poly[degree]

        _ = model.fit(x_poly, y)
        y_poly_pred = model.predict(x_poly)

        rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
        l_rms.append(rmse)
    
    anRMS = np.array(l_rms)

    iDegree = anRMS.argmin()
    anRMS[iDegree]

    # print(f"p_{iDegree}: RMS: {anRMS[iDegree]}")

    l_rms.append(anRMS[iDegree])
    l_p.append(iDegree)



polynomial_features= PolynomialFeatures(degree=iDegree)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()










###############################################################################


from scipy.interpolate import lagrange

x = np.array([0, 1, 2])

y = x**3

poly = lagrange(x, y)

poly(np.array([0, 1, 2, 3]))

