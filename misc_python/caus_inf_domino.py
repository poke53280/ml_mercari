
#
# https://blog.dominodatalab.com/understanding-causal-inference/
#


import numpy as np
import pandas as pd
 
N = 100
 
x = np.random.normal(size=N)
d = np.random.binomial(1., 0.5, size=N)
y = 3. * d + x + np.random.normal()
 
df = pd.DataFrame({'x': x, 'd': d, 'y': y})

naive_ATE = df[df.d == 1].y.mean() - df[df.d == 0].y.mean()

print(f"Naive average treatment effect (ATE) = {naive_ATE}")

from statsmodels.api import OLS

df['intercept'] = 1
model = OLS(df.y, df[['d', 'intercept']])
result = model.fit()
result.summary()


#################################################################

N = 10000
 
neighborhood = np.array(range(N))
 
industry = neighborhood % 3
 
race = ((neighborhood % 3 + np.random.binomial(3, p=0.2, size=N))) % 4

income = np.random.gamma(25, 1000*(industry + 1))

crime = np.random.gamma(100000. / income, 100, size=N)


X = pd.DataFrame({'$R$': race, '$I$': income, '$C$': crime, '$E$': industry, '$N$': neighborhood})

X.corr()

from statsmodels.api import GLM
import statsmodels.api as sm

X['$1/I$'] = 1. / X['$I$']
model = GLM(X['$C$'], X[['$1/I$']], family=sm.families.Gamma())
result = model.fit()
result.summary()


races = {0: 'african-american', 1: 'hispanic', 2: 'asian', 3: 'white'}

X['race'] = X['$R$'].map(races) 

race_dummies = pd.get_dummies(X['race']) 

X[race_dummies.columns] = race_dummies


X_restricted = X[X['$E$'] == 0]



model = OLS(X_restricted['$C$'], X_restricted[race_dummies.columns])
result = model.fit()
result.summary()



industries = {i: 'industry_{}'.format(i) for i in range(3)}


X['industry'] = X['$E$'].map(industries)

industry_dummies = pd.get_dummies(X['industry'])
X[industry_dummies.columns] = industry_dummies
x = list(industry_dummies.columns)[1:] + list(race_dummies.columns)
model = OLS(X['$C$'], X[x])
result = model.fit()
result.summary()

