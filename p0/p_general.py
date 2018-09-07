
#
# Returns series with value randomly changed +/- input number of days
#

import numpy as np
import pandas as pd
import cx_Oracle
import json

import concurrent
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import time


def Arena_to_IDX(arena_id):
    isFound_Arena_to_FID = arena_id in dict_ARENA_ID_TO_FID

    if isFound_Arena_to_FID:
        fid_str = dict_ARENA_ID_TO_FID[arena_id]

        if fid_str in dictFID_TO_IDX:
            return dictFID_TO_IDX[fid_str]
    return -1

"""c"""




def addNoise(s, num_days):

    sn = s.values
    noise = np.random.randint(-num_days, num_days +1, size = len(s))

    sn = sn + noise
    return pd.Series(sn, dtype=np.int)

"""c"""

##########################################################################
#
#   create_json_IDX_F_T_SAKSKODE
#

def create_json_IDX_F_T_SAKSKODE(df):

    d = {}

    for index, row in df.iterrows():
        id = row['IDX']
        f  = row['F']
        t  = row['T']
        s  = row['SAKSKODE']

        if id not in d:
            d[id] = { }

        if s not in d[id]:
            d[id][s] = {}
            d[id][s] = []

        d[id][s].append(f)
        d[id][s].append(t)


    return d

"""c"""

##########################################################################
#
#   create_json_IDX_F_T_SAKSKODE_VIRKID
#

def create_json_IDX_F_T_SAKSKODE_VIRKID(df):

    d = {}

    for index, row in df.iterrows():
        id = row['IDX']
        f  = row['F']
        t  = row['T']
        s  = row['SAKSKODE']
        v  = row['VIRK_ID']

        if id not in d:
            d[id] = { }

        if s not in d[id]:
            d[id][s] = {}
            d[id][s] = []

        d[id][s].append(f)
        d[id][s].append(t)
        d[id][s].append(v)


    return d

"""c"""

##########################################################################
#
#   create_json_IDX_F_F_T_SAKSKODE_DOCID_DIAG_
#
# ['DOCID', 'F0', 'F1', 'T1', 'DIAG', 'FID', 'SAKSKODE', 'IDX']

def create_json_IDX_F_F_T_SAKSKODE_DOCID_DIAG_(df):

    d = {}

    for index, row in df.iterrows():
        id = row['IDX']
        s  = row['SAKSKODE']

        if id not in d:
            d[id] = { }

        if s not in d[id]:
            d[id][s] = {}
            d[id][s] = []

        d[id][s].append(row['F0'])
        d[id][s].append(row['F1'])
        d[id][s].append(row['T1'])
        d[id][s].append(row['DIAG']) 
        d[id][s].append(row['DOCID']) 
        
    return d

"""c"""


##########################################################################
#
#   write_json
#

def write_json(filename, json_struct):
    with open(filename, 'w') as outfile:
        json.dump(json_struct, outfile, sort_keys = True, indent = 4, ensure_ascii = False)


def create_json(df):

    d = {}

    for index, row in df.iterrows():
        id = row['ID']
        f  = row['F']
        t  = row['T']
        s  = row['SAKSKODE']

        if id not in d:
            d[id] = { }

        if s not in d[id]:
            d[id][s] = {}
            d[id][s] = []

        d[id][s].append(f)
        d[id][s].append(t)


    return d

"""c"""



def serial_date_to_string(srl_no):
    new_date = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(srl_no - 1)
    return new_date.strftime("%Y-%m-%d")


"""c"""

def apply_FID_COL(df, GetFID):
    start_time = time.time()
    s = df.FK.apply(GetFID)
    print('[{}] Done FK_FID conversion'.format(time.time() - start_time))

    df = df.assign(FID=s.values)

    df_no_FID = df[(df.FID == "None")]

    nPMissing = 100 * len (df_no_FID)/ len (df)

    print(f"Missing FID on {nPMissing:.2f}% - removing...")

    df2 = df[(df.FID != "None")]

    return df2

"""c"""


##########################################################################
#
# get_gender_from_fid
#

def get_gender_from_fid(x):
    assert len(x) == 11
    g = int (x[8])
    return g % 2 == 0
    
"""c"""


##########################################################################
#
# classifyFID
#
# 'E': Error: Code is not a valid FID, D nor H number
# 'F': Valid F-number
# 'D': Valid D-number
# 'H': valid H-number
#


def classifyFID(fid):

    n = len(fid)

    if n != 11:
        return 'E'

    try:
        i = int(fid)
    except ValueError:
        return 'E'

    birth_year = int (fid[4:6])

    if birth_year > 20:
        birth_year = birth_year + 1900
    else:
        birth_year = birth_year + 2000
    
    birth_day_hi = int (fid[0])
    birth_day_lo = int (fid[1])

    number_state = 'F'
    
    if birth_day_hi >= 4:
        birth_day_hi = birth_day_hi - 4
        number_state = 'D'

    birth_day = birth_day_lo + 10 * birth_day_hi

    birth_month_hi = int (fid[2])
    birth_month_lo = int (fid[3])

    if birth_month_hi >= 4:
        if number_state == 'D':
            number_state = 'E'  # Error - cannot be both D and H
        else:
            number_state = 'H'
            birth_month_hi = birth_month_hi - 4
            print(f"Found H-nummer: {fid}")

    birth_month = birth_month_lo + 10 * birth_month_hi

    delta = 0

    try:
        delta = (pd.datetime(birth_year, birth_month, birth_day)  - pd.datetime(1970, 1, 1)).days

    except ValueError:
        print(f"Error creating date for fid = {fid}. year = {birth_year}, month = {birth_month}, day = {birth_day}")
        number_state = 'E'

    return number_state


##########################################################################
#
# toDaysSinceEpochFromFID
#
# asserts on invalid FID.
# Run classifyFID first and get rid of bad data before this conversion.
#


def toDaysSinceEpochFromFID(fid):

    n = len(fid)

    assert n== 11, f"Error {fid} - len = {len(fid)}"

    birth_year = int (fid[4:6])

    if birth_year > 20:
        birth_year = birth_year + 1900
    else:
        birth_year = birth_year + 2000
    
    birth_day_hi = int (fid[0])
    birth_day_lo = int (fid[1])
    
    
    isD = False

    if birth_day_hi >= 4:
        birth_day_hi = birth_day_hi - 4
        isD = True


    birth_day = birth_day_lo + 10 * birth_day_hi

    birth_month_hi = int (fid[2])
    birth_month_lo = int (fid[3])

    if birth_month_hi >= 4:
        assert not isD, f"Both D and H number: {fid}"
        birth_month_hi = birth_month_hi - 4
        print(f"Found H-nummer: {fid}")
        

    birth_month = birth_month_lo + 10 * birth_month_hi
    
    delta = (pd.datetime(birth_year, birth_month, birth_day)  - pd.datetime(1970, 1, 1)).days
    # Let throw ValueError

    return delta



def toDaysSinceEpoch(x):
    x = pd.to_datetime(x)
    return (x - pd.datetime(1970, 1, 1)).days

"""c"""

def toDateTimeFromEpoch(x):
    x = pd.datetime(1970, 1, 1) + timedelta(days=x)
    return x

##########################################################################
#
#   get_chunk_end
#

def get_chunk_end(df, start_idx, colname) :

    if (start_idx >= len (df)):
        return len(df)

    id = df.iloc[start_idx][colname]
    idx = start_idx + 1

    while idx < len (df):
        isIDEqual = df.iloc[idx][colname] == id

        if (isIDEqual):
            pass
        else:
            break
    
        idx = idx + 1

    return idx

w = 90

##########################################################################
#
#   get_chunks
#

def get_chunks(df, size, colname):

    start = 0
    end = 0

    l = []

    while (end < len(df)):

        end = get_chunk_end(df, start + size, colname)

        l.append( (start, end ))

        start = end

    return l








