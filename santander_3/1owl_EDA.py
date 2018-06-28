

import xgboost as xgb
import lightgbm as lgb
from sklearn import *
import pandas as pd
import numpy as np




d = DReduction()

d.fit(train)

df_train6 = d.transform(train)
df_test6 = d.transform(test)


train = pd.concat([train, df_train6], axis = 1)
test = pd.concat([test, df_test6], axis = 1)



