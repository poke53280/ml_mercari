
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sklearn.datasets
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

sns.set()

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 1500)


def run_linear_regression(X, y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    y_test_predict = lin_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    return rmse


def get_best_addition(df, best_rmse, best_cols, test_cols):
    best_in_experiment = best_rmse

    best_column = ''

    for c in test_cols:

        X = df[best_cols + [c]].values
        rmse = run_linear_regression(X, y)
        print(f"With {c}: RMSE = {rmse} ({best_rmse})")

        if rmse < best_in_experiment:
            best_in_experiment = rmse
            best_column = c

    return best_column, best_in_experiment



boston_dataset = sklearn.datasets.load_boston()

df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

y = boston_dataset.target


correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)

plt.show()

l_columns = list (df.columns)

# l_cols = ['LSTAT', 'RM']

l_res = []

for x in range (10000):

    l_cols = np.random.choice(l_columns, size = 1, replace = False)

    X = df[l_cols].values

    rmse = run_linear_regression(X, y)

    t = (l_cols[0], rmse)

    l_res.append(t)

df_res = pd.DataFrame(l_res)

df_res.columns = ['a', 'rmse']
df_res = df_res.sort_values(by = 'rmse')


# Best:
b = df_res.iloc[0]

best_cols = [b.a, b.b, b.c]

l_columns = df.columns

test_cols = list (set(l_columns) - set (best_cols))

best_rmse = b.rmse

while len(test_cols) > 0:

    c, rms = get_best_addition(df, best_rmse, best_cols, test_cols)

    print (c, rms)

    if rms < best_rmse:
        print(f"New record {rms}")
        best_rmse = rms
        best_cols = best_cols + [c]
        test_cols = list (set(l_columns) - set (best_cols))
    else:
        break
    

