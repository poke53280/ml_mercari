

# Configuration parameters

CONFIG_DATA_START = "2006-01-01"
CONFIG_DATA_INVALID_FUTURE = "2019-01-01"

# Resolving environment

import os

local_dir = os.getenv('LOCAL_PY_DIR')
assert local_dir is not None, "Set environment variable LOCAL_DIR to location of this python file."
print(f"Local python directoy is set to {local_dir}")

os.chdir(local_dir)
config_file = os.getenv('DB_DATA')
assert config_file is not None, "No config file found in environment variable DB_DATA"
print(f"Using database connection file '{config_file}'")



import sys
import pandas as pd
import json
import random
import numpy as np
import bisect
import datetime
import time


from p_general import DataProvider
from p_general import IDConverter
from p_general import toDaysSinceEpoch
from p_general import apply_FID_COL
from p_general import classifyFID
from p_general import toDaysSinceEpochFromFID
from p_general import get_gender_from_fid

os.environ['NLS_NCHAR_CHARACTERSET']='AL16UTF16'
os.environ['NLS_CHARACTERSET']='WE8ISO8859P15'
os.environ['NLS_LANG']='AMERICAN_AMERICA.WE8ISO8859P15'


f = open(config_file,"r")
config = f.read()
f.close()

dp = DataProvider(config)

l_queries = []

l_queries.append( ("A", "sql_syk3",      "syk") )
#l_queries.append( ("A", "sql_fravar",    "fravar") )
l_queries.append( ("A", "sql_pmap",      "pmap") )
#l_queries.append( ("B", "sql_vedtak",    "vedtak") )
#l_queries.append( ("B", "sql_meldekort", "meldekort") )
#l_queries.append( ("C", "select_large",  "aa") )

d = dp.async_load(l_queries)

p = d['pmap'].copy()

p.columns = ['FID', 'FK', 'A']

m = p.FK == -1

p = p[~m]

# Clean up FID to FK mapping.

p_state = p["FID"].apply(classifyFID)

m = p_state == 'E'
p = p[~m]

p_state = p["FID"].apply(classifyFID)
p_epoch = p["FID"].apply(toDaysSinceEpochFromFID)

p = p.assign(S = p_state)
p = p.assign(E = p_epoch)


# Prepare dictionaries: FK -> FID, FK -> epoch birth, FK -> State

l_fid = p.FID.tolist()
l_fk  = p.FK.tolist()
l_a   = p.A.tolist()
l_s   = p.S.tolist()
l_e   = p.E.tolist()


d_FK_TO_FID = dict (zip (l_fk, l_fid))
d_FK_TO_E = dict (zip (l_fk, l_e))
d_FK_TO_S = dict (zip (l_fk, l_s))




def isExists(x, d):
    return x in d



df_syk = d['syk'].copy()

df_syk.columns = ["FK", "DID", "F0", "F1", "T0", "D"]


# Data stream start time as documented in project DPIA. Discard earlier data.

syk_data_begin = toDaysSinceEpoch(CONFIG_DATA_START)

m = (df_syk.F0 >= syk_data_begin)

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data before {CONFIG_DATA_START}. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]


syk_data_future = toDaysSinceEpoch(CONFIG_DATA_INVALID_FUTURE)

m = (df_syk.F0 < syk_data_future) & (df_syk.F1 < syk_data_future) & (df_syk.T0 < syk_data_future)

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data after {CONFIG_DATA_INVALID_FUTURE}. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]

m = df_syk.FK.apply(isExists, args = (d_FK_TO_FID,))

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data where no FID <-> FK map found. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]

def getValue(x, d):
    return d[x]

fid = df_syk.FK.apply(getValue, args = (d_FK_TO_FID,))
birth = df_syk.FK.apply(getValue, args = (d_FK_TO_E,))
fid_s = df_syk.FK.apply(getValue, args = (d_FK_TO_S,))

df_syk = df_syk.assign(B = birth)
df_syk = df_syk.assign(FID = fid)
df_syk = df_syk.assign(FID_S = fid_s)

g = df_syk["FID"].apply(get_gender_from_fid)

df_syk = df_syk.assign(G = g)

q = df_syk.FID.astype('category').cat.codes

df_syk = df_syk.assign (IDX = q)

q = df_syk.DID.astype('category').cat.codes

df_syk = df_syk.assign (MD = q)

df_syk = df_syk.drop(["FK", "FID", "DID"], axis = 1)

df_syk = df_syk.assign(D = df_syk.D.astype('category'))
df_syk = df_syk.assign(FID_S = df_syk.FID_S.astype('category'))




# Test - how many new in a small range:

syk_entry_begin = toDaysSinceEpoch("2014-10-01")
syk_entry_end = toDaysSinceEpoch("2014-10-02")

m = (df_syk.F0 >= syk_entry_begin) & (df_syk.F0 < syk_entry_end)
q = df_syk[m]


df = df_syk

# Offset all dates into positive numbers for convenience.

first_day = np.min(  [df.F0.min(), df.F1.min(), df.T0.min(), df.B.min()])

print(f"All days offset with {first_day} day(s)")


df.F0 -= first_day
df.F1 -= first_day
df.T0 -= first_day
df.B -= first_day


min_day = np.min(  [df.F0.min(), df.F1.min(), df.T0.min(), df.B.min()])

assert min_day == 0

max_day = np.max(  [df.F0.max(), df.F1.max(), df.T0.max(), df.B.max()])


def downcast_unsigned(s):
    assert s.min() >= 0
    return pd.to_numeric(s, downcast = 'unsigned')


# Convert data types to compact forms

df = df.assign(MD = downcast_unsigned(df.MD))
df = df.assign(IDX = downcast_unsigned(df.IDX))

df = df.assign(F0 = downcast_unsigned(df.F0))
df = df.assign(F1 = downcast_unsigned(df.F1))
df = df.assign(T0 = downcast_unsigned(df.T0))
df = df.assign(B = downcast_unsigned(df.B))

q = df_syk.FID_S.astype('category').cat.codes

df = df.assign(FID_S = q)
df = df.assign(FID_S = downcast_unsigned(df.FID_S))


# Diagnose FE
q = df.D.apply(len)

df = df.assign(D_L = q)

q = df.D.apply(lambda x: x[0])

df = df.assign(D_H = q)

q = df.D.apply(lambda x: x[1:])

df = df.assign(D_C = q)

df = df.assign(D_H = df.D_H.astype('category'))
df = df.assign(D_C = df.D_C.astype('category'))

q = df.D_H.cat.codes
df = df.assign(D_H = q)

q = df.D_C.cat.codes
df = df.assign(D_C = q)

q = df.D.cat.codes
df = df.assign(D = q)

df = df.assign(D = downcast_unsigned(df.D))
df = df.assign(G = df.G.astype('uint8'))

df = df.assign(IDX =  downcast_unsigned(df.IDX))

df = df.assign(D_L = downcast_unsigned(df.D_L))

df = df.assign(D_H = downcast_unsigned(df.D_H))

df = df.assign(D_C = downcast_unsigned(df.D_C))

# No value in D now
df = df.drop(['D'], axis = 1)


df = df.sort_values(by = ['IDX', 'F0', 'F1', 'T0'])

df = df.reset_index(drop=True)

df.to_pickle("S:\\F0326\\Grupper\\Gruppe18\\tmp.pkl")

#####################################################################################








