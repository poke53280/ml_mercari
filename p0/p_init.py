

import sys
import pandas as pd
import os

print(sys.version)

import json
import random

import numpy as np
import bisect

import datetime
import time


os.environ['NLS_NCHAR_CHARACTERSET']='AL16UTF16'
os.environ['NLS_CHARACTERSET']='WE8ISO8859P15'
os.environ['NLS_LANG']='AMERICAN_AMERICA.WE8ISO8859P15'

# Configuration parameters

CONFIG_DATA_START = "2006-01-01"



# ---- Read in configuration file and create json dictionary.

config_file = os.getenv('DB_DATA')

assert config_file is not None, "No config file found in environment variable DB_DATA"

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


p = d['pmap']

p.columns = ['FID', 'FK', 'A']

id_converter = IDConverter(p)


#### Match sykefravaer and fid ###################

df_syk = d['syk']  # Ref, not copy

df_syk.head()

df_syk.columns = ["FK", "DID", "F0", "F1", "T0", "D"]


# Data stream start time as documented in project DPIA. Discard earlier data.



syk_data_begin = toDaysSinceEpoch(CONFIG_DATA_START)

m = (df_syk.F0 >= syk_data_begin)

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data before {CONFIG_DATA_START}. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]

# Is here a mapping to FID in all cases?

q = df_syk.FK.apply(lambda x: not id_converter.GetFID_FROM_FK(x) == "None")

q.value_counts()

# Note: Removing the 3% or so persons without map FK->FID

df_syk = apply_FID_COL(df_syk, id_converter.GetFID_FROM_FK)


# Get birth day with noise
# Todo: Expose noise parameter

q = df_syk["FID"].apply(get_random_epoch_birth_day)

df_syk = df_syk.assign(B = q)

g = df_syk["FID"].apply(get_gender_from_fid)

df_syk = df_syk.assign(G = g)


q = df_syk.FID.astype('category').cat.codes

df_syk = df_syk.assign (IDX = q)

q = df_syk.DID.astype('category').cat.codes

df_syk = df_syk.assign (MD = q)

df_syk = df_syk.drop(["FK", "FID", "DID"], axis = 1)

df_syk = df_syk.assign(D = df_syk.D.astype('category'))


# Todo: Add noise to shift together.
#df_syk.F0 = addNoise(df_syk.F0, 3)
#df_syk.F1 = addNoise(df_syk.F1, 3)
#df_syk.T1 = addNoise(df_syk.T1, 3)


# Test - how many new in a small range:

syk_entry_begin = toDaysSinceEpoch("2014-10-01")
syk_entry_end = toDaysSinceEpoch("2014-10-02")

m = (df_syk.F0 >= syk_entry_begin) & (df_syk.F0 < syk_entry_end)
q = df_syk[m]






