

import pandas as pd
import numpy as np


DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


df = train = pd.read_csv(DATA_DIR + 'noised_intervals.csv')


df.columns

df.columns = ['drop', 'begin', 'end', 'P']

df = df.drop(['drop'], axis = 1)

df = df[['P', 'begin', 'end']]

df = df.sort_values(by = ['P', 'begin'])










