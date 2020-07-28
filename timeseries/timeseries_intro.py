

# https://towardsdatascience.com/inferring-causality-in-time-series-data-b8b75fe52c46




import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt


input = np.array([3, 4, 1, 2, 3, 4, 5, 14, 12, 3])

num_data = input.shape[0]


df_dvh = pd.read_pickle("C:\\Users\\T149900\\ts_data\\sok_inn.pkl")
df_dvh = df_dvh.rename(columns = {'dn_count': 'd_time', 'inngang_prior': 'y'})


df_sm = pd.read_pickle("C:\\Users\\T149900\\ts_data\\sm_inn.pkl")

# CONTINUE HERE

# Plot 


plt.plot(df_dvh.d_time, df_dvh.y)
plt.plot(df_sm.index.values, df_sm.Alle)

plt.show()




def get_data_row(df_stat, t):

    num_rows = 16
    dn_age = 2

    end_date =  t - dn_age
    start_date = end_date - num_rows

    m = (df_stat.d_time >= start_date) & (df_stat.d_time <= end_date)

    return np.concatenate([ df_stat[m].Alle.values, df_stat[m].R99x_new.values] )


l = []
l_t = []


for t in range (18310, 18349):
    l.append(get_data_row(df_stat, t))
    l_t.append(t)


df = pd.DataFrame(l)

df = df.assign(d_time = pd.Series(l_t))

df = df.merge(right = df_dvh, on = 'd_time', how = 'left')

df = df.drop('d_time', axis = 1)

data = np.asmatrix(df)

y0 = np.array(data[:, -1]).squeeze()

X0 = data[:, :-1]

#assert (X0[:, 8] == 0).sum() == 0
#X0[:, 9] = (y0 * np.array(X0[:, 8]).squeeze())[:, np.newaxis]


eps = 1e-12

X = np.log(X0 + eps)
y = np.log(y0 + eps)



polynomial_features= PolynomialFeatures(degree=2)

X_poly = polynomial_features.fit_transform(X)


X_train = X_poly[:33]
X_valid = X_poly[33:]

y_train = y[:33]
y_valid = y[33:]



clf = Lasso(alpha = 0.15, max_iter = 50000)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid).squeeze()
y_pred_0 = np.exp(y_pred) - eps

y_pred_0.astype(int)

y0[33:]

clf.coef_, clf.intercept_




import tensorflow as tf
import tensorflow_probability as tfp

tf.test.gpu_device_name()


trend = tfp.sts.LocalLinearTrend(observed_time_series=co2_by_month)



seasonal = tfp.sts.Seasonal(
    num_seasons=12, observed_time_series=co2_by_month)
model = tfp.sts.Sum([trend, seasonal], observed_time_series=co2_by_month)

