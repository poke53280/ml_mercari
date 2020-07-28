


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import operator

data =[   0.        ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        , 5104.71428571, 5090.71428571, 4519.42857143, 4273.14285714, 4175.14285714, 4206.71428571,
       4235.57142857, 4260.28571429, 4306.42857143, 4390.        , 4415.42857143, 4374.28571429, 4216.28571429, 4050.71428571, 4031.71428571, 3997.42857143, 3892.42857143, 3846.85714286,
       3908.71428571, 4094.28571429, 4366.14285714, 4367.85714286, 4368.42857143]


num_days_look_back = 20

x = np.array(data)

num_data = x.shape[0]

t_min = 0

t = np.array(list (range(0, num_data)))


t_pred = np.array(list (range(t_min + num_data - 10, t_min + num_data + 12)))

# Last num_days_look_back only for train:
t_train = t[-num_days_look_back:]
x_train = x[-num_days_look_back:]

t_train = t_train[:, np.newaxis]
x_train = x_train[:, np.newaxis]


t_all = t[:, np.newaxis]
x_all = x[:, np.newaxis]




t_pred = t_pred[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree=1)
t_poly_train = polynomial_features.fit_transform(t_train)

t_poly_pred = polynomial_features.fit_transform(t_pred)


model = LinearRegression()
model.fit(t_poly_train, x_train)


x_pred = model.predict(t_poly_pred)


plt.scatter(t_all, x_all, s = 10)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(t_pred,x_pred), key=sort_axis)
t_out, x_poly_pred = zip(*sorted_zip)
plt.plot(t_out, x_poly_pred, color='m')

plt.ylim(bottom= -10) 

plt.show()

def get_datetime(nEpochDays):
    return datetime.datetime(1970,1,1,0,0) + datetime.timedelta(nEpochDays)



