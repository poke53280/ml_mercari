

import sys
import pandas as pd
import os

print(sys.version)
import cx_Oracle
import getpass
import json
import random

import numpy as np
import bisect

os.environ['NLS_NCHAR_CHARACTERSET']='AL16UTF16'
os.environ['NLS_CHARACTERSET']='WE8ISO8859P15'
os.environ['NLS_LANG']='AMERICAN_AMERICA.WE8ISO8859P15'

import datetime
import time

import h5py   

def GetDaysSinceEpoch(s, epoch, na_date):
    # From string to datetime, set NaT at error
    s_out = pd.to_datetime(s, errors = 'coerce')

    # Set NA to 2019, assume still active in job.
    s_out = s_out.fillna(future)

    # Offset from UNIX epoch
    s_out = s_out - epoch

    # Fast conversions to int.

    s_out = s_out.astype('timedelta64[D]')
    s_out = s_out.astype(int)

    return s_out


def GetDaysSinceEpoch(s, epoch, na_date):
    # From string to datetime, set NaT at error
    s_out = pd.to_datetime(s, errors = 'coerce')

    # Set NA to 2019, assume still active in job.
    s_out = s_out.fillna(future)

    # Offset from UNIX epoch
    s_out = s_out - epoch

    # Fast conversions to int.

    s_out = s_out.astype('timedelta64[D]')
    s_out = s_out.astype(int)

    return s_out


#############################################################################

def addFID(fid):
    if fid in d:
        pass
    else:
        value = len(d)
        d[fid] = value

        
def getSeqFromFID(fid):
    assert (fid in d)
    return d[fid]        
        
########################################################################
#
#   get_birth_year_from_fid
#

def get_birth_year_from_fid(fid):
    if len(fid) == 11:
        try:
            b = fid[4:6]
            return int (b)
        except ValueError:
            print(f"Warning bad fid: {fid}")
            return -1

    else:
        return -1




##########################################################################
#
# get_random_epoch_birth_day
#

def get_random_epoch_birth_day(fid):
    assert (len(fid) == 11)

    birth_year = int (fid[4:6])

    if birth_year > 20:
        birth_year = birth_year + 1900
    else:
        birth_year = birth_year + 2000

    start_date = datetime.date(day=1, month=1, year=birth_year).toordinal()

    end_date = datetime.date(day=31, month=12, year=birth_year).toordinal()

    random_day = datetime.date.fromordinal(random.randint(start_date, end_date))

    epoch_day = datetime.date(1970, 1, 1)

    days_since_epocy = (random_day - epoch_day).days 

    return days_since_epocy


w = 90


##########################################################################
#
#   get_random_epoch_birth_day_from_birth_year
#

def get_random_epoch_birth_day_from_birth_year(epoch_day, birth_year):

    if birth_year > 20:
        birth_year = birth_year + 1900
    else:
        birth_year = birth_year + 2000

    start_date = datetime.date(day=1, month=1, year=birth_year).toordinal()
    end_date   = datetime.date(day=31, month=12, year=birth_year).toordinal()

    random_day = datetime.date.fromordinal(random.randint(start_date, end_date))

    days_since_epocy = (random_day - epoch_day).days 

    return days_since_epocy

"""c"""

###############################################################################
#
#   GetTrueAndFalse
#
#   Returns series with value randomly changed +/- input number of days
#

def GetTrueAndFalse(q):

    nCount = len (q)

    w = q.value_counts()

    counts = w.tolist()
    values = w.index.tolist()

    d = dict (zip (values, counts))

    if True in d:
        nTrue = d[True]
        nFalse = nCount - nTrue
    else:
        nFalse = nCount
        nTrue = 0

    return {'True':nTrue, 'False':nFalse}

"""c"""


###############################################################################
#
#   addNoise
#
#   Returns numpy array with value randomly changed +/- input number of days
#

def addNoise(s, num_days):

    null_count = s.isnull().sum()
    assert (null_count == 0)

    sn = s.values
    noise = np.random.randint(-num_days, num_days +1, size = len(s))

    sn = sn + noise
    return sn

"""c"""


#
# Returns -1 if not found
#

def Arena_to_IDX(arena_id):
    isFound_Arena_to_FID = arena_id in dict_ARENA_ID_TO_FID

>>>>>>> 0ece48254089977a6525f0b207088ea052d6e201
    if isFound_Arena_to_FID:
        fid_str = dict_ARENA_ID_TO_FID[arena_id]

        if fid_str in dictFID_TO_IDX:
            return dictFID_TO_IDX[fid_str]
    return -1


#
# Returns -1 if not found
#

def Arena_to_IDX(arena_id):
    isFound_Arena_to_FID = arena_id in dict_ARENA_ID_TO_FID


##########################################################################
#
#   create_json
#

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

a = 9323

##########################################################################
#
#   write_json
#

def write_json(filename, json_struct):
    with open(filename, 'w') as outfile:
        json.dump(json_struct, outfile, sort_keys = True, indent = 4, ensure_ascii = False)


w = 90

import datetime

def serial_date_to_string(srl_no):
    new_date = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(srl_no - 1)
    return new_date.strftime("%Y-%m-%d")



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

def write_json(filename, json_struct):
    with open(filename, 'w') as outfile:
        json.dump(json_struct, outfile, sort_keys = True, indent = 4, ensure_ascii = False)

"""c"""


def serial_date_to_string(srl_no):
    new_date = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(srl_no - 1)
    return new_date.strftime("%Y-%m-%d")


"""c"""


def serial_date_to_string(srl_no):
    new_date = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(srl_no - 1)
    return new_date.strftime("%Y-%m-%d")


"""c"""


##########################################################################
#
#   getConnection
#

def getConnection(key):
    db_conf = config_data[key]

    con_string      = db_conf['connection']
    con_user        = db_conf['user']
    con_password    = db_conf['password']

    c = cx_Oracle.connect(user=con_user, password=con_password, dsn=con_string)

    print(key + ": " + c.version)
    return c

"""c"""

##########################################################################
#
#   getQuery
#

def getQuery(key, queryKey):
    query_string = ""
    try:
        db_conf = config_data[key]
        query_string = db_conf[queryKey]

    except KeyError:
        print ("Key(s) not found : '" + key + "' -> '" + queryKey + "'")
        pass

    assert len(query_string) > 0

    if type(query_string) is list:
        query_string = " ".join(query_string)

    return query_string


"""c"""

#####################################################################
#
#   mem_info_MB
#
#

def mem_info_MB(df):
    mem = df.memory_usage(index=True, deep=True).sum()
    return mem/ 1000 / 1000

"""c"""

#####################################################################
#
#   mem_info_MB
#
#

def mem_info_MB(df):
    mem = df.memory_usage(index=True).sum()
    return mem/ 1000 / 1000

"""c"""


#####################################################################
#
#   mem_info_MB
#
#

def mem_info_MB(df):
    mem = df.memory_usage(index=True).sum()
    return mem/ 1000 / 1000

"""c"""



#####################################################################
#
#   report_bandwidth
#
#

def report_bandwidth(desc, start_time, end_time, df):
    
    processing_time = end_time - start_time
    memMB = mem_info_MB (df)
    rBandwidth = memMB / processing_time

    print(f"[Extraction: {desc} [Time: {processing_time:.2f}s] [Data: {memMB:.1f} MB] [Bandwidth: {rBandwidth:.2f} MB/s]")

"""c"""

#####################################################################
#
#   report_bandwidth
#
#

def report_bandwidth(start_time, end_time, df):
    
    processing_time = end_time - start_time
    memMB = mem_info_MB (df)
    rBandwidth = memMB / processing_time

    print(f"[Time: {processing_time:.2f}s] [Data: {memMB:.1f} MB] [Bandwidth: {rBandwidth:.2f} MB/s]")

"""c"""


#####################################################################
#
#   report_bandwidth
#
#

def report_bandwidth(start_time, end_time, df):
    
    processing_time = end_time - start_time
    memMB = mem_info_MB (df)
    rBandwidth = memMB / processing_time

    print(f"[Time: {processing_time:.2f}s] [Data: {memMB:.1f} MB] [Bandwidth: {rBandwidth:.2f} MB/s]")

"""c"""





#####################################################################
#
#   read_df
#
#

def read_df (db_name, db_query):
    conn = getConnection(db_name)
    sql = getQuery(db_name, db_query)
    
    start_time = time.time()

    cur = conn.cursor()

    cur.arraysize = 100000

    my_exec = cur.execute(sql)

    results = my_exec.fetchall()

    my_exec.close()

    end_time = time.time()

    conn.close()

    df = pd.DataFrame(results)

    report_bandwidth (db_name + " " + db_query + ": ", start_time, end_time, df)

    return df

"""c"""


#####################################################################
#
#   read_df
#
#

def read_df (db_name, db_query):
    conn = getConnection(db_name)
    sql = getQuery(db_name, db_query)
    
    start_time = time.time()

    cur = conn.cursor()

    cur.arraysize = 100000

    my_exec = cur.execute(sql)

    results = my_exec.fetchall()

    my_exec.close()

    end_time = time.time()

    conn.close()

    df = pd.DataFrame(results)

    report_bandwidth(start_time, end_time, df)

    return df

"""c"""

#####################################################################
#
#   read_df
#
#

def read_df (db_name, db_query):
    conn = getConnection(db_name)
    sql = getQuery(db_name, db_query)
    
    start_time = time.time()

    cur = conn.cursor()

    cur.arraysize = 100000

    my_exec = cur.execute(sql)

    results = my_exec.fetchall()

    my_exec.close()

    end_time = time.time()

    conn.close()

    df = pd.DataFrame(results)

    report_bandwidth(start_time, end_time, df)

    return df
=======


#####################################################################
#
#   mem_info_MB
#
#

def mem_info_MB(df):
    mem = df.memory_usage(index=True).sum()
    return mem/ 1000 / 1000

"""c"""

#####################################################################
#
#   report_bandwidth
#
#

def report_bandwidth(start_time, end_time, df):
    
    processing_time = end_time - start_time
    memMB = mem_info_MB (df)
    rBandwidth = memMB / processing_time

    print(f"[Time: {processing_time:.2f}s] [Data: {memMB:.1f} MB] [Bandwidth: {rBandwidth:.2f} MB/s]")

"""c"""

#####################################################################
#
#   read_df
#
#

def read_df (db_name, db_query):
    conn = getConnection(db_name)
    sql = getQuery(db_name, db_query)
    
    start_time = time.time()

    cur = conn.cursor()

    cur.arraysize = 100000

    my_exec = cur.execute(sql)

    results = my_exec.fetchall()

    my_exec.close()

    end_time = time.time()

    conn.close()

    df = pd.DataFrame(results)

    report_bandwidth(start_time, end_time, df)

    return df

"""c"""



#########################################################################
#
#   db_load
#

def db_load(isAlna):

    d = {}

    start_time = time.time()

    d['syk'] = read_df("A", "sql_syk3")

    d['fravar'] = read_df("A", "sql_fravar")

    d['pmap'] = read_df("A", "sql_pmap")

    d['vedtak'] = read_df("B", "sql_vedtak")

    d['meldekort'] = read_df("B", "sql_meldekort")

    d['aa'] = read_df("C", "select_large")

    if isAlna:
        d['alna_pop'] = read_df("B", "sql_population")

    end_time = time.time()

    processing_time = end_time - start_time

    print(f"Database load compete. Time: {processing_time:.2f}s]")

    return d

"""c"""


#########################################################################
#
#   db_load
#

def db_load(isAlna):

    d = {}

    start_time = time.time()

    d['syk'] = read_df("A", "sql_syk3")

    d['fravar'] = read_df("A", "sql_fravar")

    d['pmap'] = read_df("A", "sql_pmap")

    d['vedtak'] = read_df("B", "sql_vedtak")

    d['meldekort'] = read_df("B", "sql_meldekort")

    d['aa'] = read_df("C", "select_large")

    if isAlna:
        d['alna_pop'] = read_df("B", "sql_population")

    end_time = time.time()

    processing_time = end_time - start_time
=======
    d = {}

    start_time = time.time()

    d['syk'] = read_df("A", "sql_syk3")

    d['fravar'] = read_df("A", "sql_fravar")

    d['pmap'] = read_df("A", "sql_pmap")

    d['vedtak'] = read_df("B", "sql_vedtak")

    d['meldekort'] = read_df("B", "sql_meldekort")

    d['aa'] = read_df("C", "select_large")

    if isAlna:
        d['alna_pop'] = read_df("B", "sql_population")

    end_time = time.time()

    processing_time = end_time - start_time

    print(f"Database load compete. Time: {processing_time:.2f}s]")

    return d
>>>>>>> a26b7b55ebf0dd2cda569dfd77d417b9de45ff94

    print(f"Database load compete. Time: {processing_time:.2f}s]")

<<<<<<< HEAD
    return d

"""c"""

=======
>>>>>>> a26b7b55ebf0dd2cda569dfd77d417b9de45ff94





#########################################################################
#
#   db_load
#

def db_load(isAlna):

    d = {}

    start_time = time.time()


    d['enhet'] ? read df enhet

    d['syk'] = read_df("A", "sql_syk3")

    d['fravar'] = read_df("A", "sql_fravar")

    d['pmap'] = read_df("A", "sql_pmap")

    d['vedtak'] = read_df("B", "sql_vedtak")

    d['meldekort'] = read_df("B", "sql_meldekort")

    d['aa'] = read_df("C", "select_large")

    if isAlna:
        d['alna_pop'] = read_df("B", "sql_population")

    end_time = time.time()

    processing_time = (end_time - start_time) / 60

    print(f"Database load compete. Time: {processing_time:.1f}mins]")

    return d

"""c"""


def preprocess_aa(df_aa):

    future = pd.datetime(2019, 1, 1)
    epoch  = pd.datetime(1970, 1, 1)

    df_aa.columns = ['FID', 'FRA', 'TIL']

    sFRA = GetDaysSinceEpoch(df_aa.FRA, epoch, future)
    sTIL = GetDaysSinceEpoch(df_aa.TIL, epoch, future)

    df_aa = df_aa.assign(FRA = sFRA)
    df_aa = df_aa.assign(TIL = sTIL)
=======

#
# MELDEKORT
#

df_meldekort = d_all['meldekort']

def preprocess_meldekort (df_meldekort):

    future = pd.datetime(2019, 1, 1)
    epoch  = pd.datetime(1970, 1, 1)

    df_meldekort.isnull().sum()
    df_meldekort.dropna(inplace = True)

    MAXDAY = (future - epoch).days

    m = df_meldekort[1] >= MAXDAY - 14

    m.value_counts()

    # Remove high values (few).

    df_meldekort = df_meldekort[~m]

    s = df_meldekort[1] + 14

    df_meldekort.columns = ['FID', 'FRA']

    df_meldekort = df_meldekort.assign(TIL = s)

    df_meldekort['SAKSKODE'] = 'MELD'

"""c"""

#
# MELDEKORT
#

df_meldekort = d_all['meldekort']

def preprocess_meldekort (df_meldekort):

    df_meldekort.isnull().sum()
    df_meldekort.dropna(inplace = True)

    MAXDAY = (future - epoch).days

    m = df_meldekort[1] >= MAXDAY - 14

    # Remove high values (few).

    df_meldekort = df_meldekort[~m]
>>>>>>> 0ece48254089977a6525f0b207088ea052d6e201

    s = df_meldekort[1] + 14

<<<<<<< HEAD
df_vedtak.head()
s = addNoise(df_vedtak.FRA, 3)
t = addNoise(df_vedtak.TIL, 3)
df_vedtak = df_vedtak.assign(FRA = s)
df_vedtak = df_vedtak.assign(TIL = t)

df_meldekort.head()
s = addNoise(df_meldekort.FRA, 1)
t = addNoise(df_meldekort.TIL, 1)
df_meldekort = df_meldekort.assign(FRA = s)
df_meldekort = df_meldekort.assign(TIL = t)

#No noise on these atm
df_syk.head()
df_fravar.head()

df_p = p
df_p.head()

"""c"""




#
# VEDTAK
#

df_vedtak = d_all['vedtak']

def preprocess_vedtak(df_vedtak):

    df_vedtak.columns = ["FID", "A_ID", "F", "T", "C"]

    df_vedtak.isnull().sum()

    m = df_vedtak.FID.isnull()

    m.value_counts()

    # Drop where FID is null

    df_vedtak = df_vedtak[~m]

    # Drop Arena ID
    df_vedtak.drop(['A_ID'], inplace = True, axis = 1)

    m = df_vedtak.FRA > df_vedtak.TIL

    #WARNING: MANY (8 %)
    GetTrueAndFalse(m)

    df_vedtak = df_vedtak[~m]

    # CLIP TO MAXDAY

    df_vedtak.FRA = df_vedtak.FRA.clip_upper(MAXDAY)
    df_vedtak.TIL = df_vedtak.TIL.clip_upper(MAXDAY)

    MINDAY = (pd.datetime(1980, 1, 1) - epoch).days

    df_vedtak.FRA = df_vedtak.FRA.clip_lower(MINDAY)
    df_vedtak.TIL = df_vedtak.TIL.clip_lower(MINDAY)

"""c"""






############################################################################
#
#   preprocess_syk
#

df_syk = d_all['syk']

def preprocess_syk(df_syk):
    df_syk.columns = ["FK", "DID", "F0", "F1", "T0", "D"]

    df_syk = id_converter.Add_FID_FROM_FK_Delete_Missing(df_syk)

    df_syk['C'] = 'SYKM'

    return df_syk

"""c"""

##############################################################
#
#   preprocess_fravar
#

df_fravar = d_all['fravar']

def preprocess_fravar(df_fravar):
    
    df_fravar.columns = ['FK', 'VID', 'F', 'T']

    df_fravar = id_converter.Add_FID_FROM_FK_Delete_Missing(df_fravar)

    df_fravar['SAKSKODE'] = 'SYKF'

    return df_fravar

"""c"""

############################################################################
#
#   preprocess_syk
#

def preprocess_syk(df_syk):
    df_syk.columns = ["FK", "DID", "F0", "F1", "T0", "D"]

    df_syk = id_converter.Add_FID_FROM_FK_Delete_Missing(df_syk)

    df_syk['SAKSKODE'] = 'SYKM'

    return df_syk

"""c"""

##############################################################
#
#   preprocess_fravar
#

def preprocess_fravar(df_fravar):
    
    df_fravar.columns = ['FK', 'VIRKID', 'F', 'T']

    df_fravar = id_converter.Add_FID_FROM_FK_Delete_Missing(df_fravar)

    df_fravar['SAKSKODE'] = 'SYKF'

    return df_fravar

"""c"""

##########################################################################
#
#   toTupleFromNumberList
#

def toTupleFromNumberList(l):

    data = []

    idx = 0

    for x in l:
       
        t = (x, idx)
        data.append(t)
        idx = idx + 1

    data.sort(key=lambda tup: tup[0])
    return data

"""c"""



##################################################################################
#
#   toTupleFromStringList
#

def toTupleFromStringList(l):

    data = []

    idx = 0

    for x in l:

        number = -1

        try:
            number = int(x)
            t = (number, idx)
            data.append(t)

        except ValueError:
            pass

        idx = idx + 1

    data.sort(key=lambda tup: tup[0])
    return data

"""c"""


##########################################################################
#
#   merge_unique_valued_lists
#

def merge_unique_valued_lists(values, ids, l_b):

    idx = []

    begin_scan = 0

    for b in l_b:
        i = bisect.bisect_left(values, b, lo = begin_scan, hi = len (values))

        if i != len(values) and values[i] == b:
            idx.append(ids[i])
            begin_scan = i +1
            if begin_scan >= len(values):
                print("Out of search space. breaking out")                
                break

    return idx

"""c"""
############################################################################

dr = range(0, 1000000)
l_a = random.sample(r, 30000)
rInA = range(0, len(l_a))
l_idx = random.sample(rInA, 23000)
l_b = [l_a[i] for i in l_idx]


# in: unsorted list. need to keep original indices.

# Add index as second tuple element


l = [3, 4, 11, 8, 1]

t = toTupleFromNumberList(l_a)

values, ids = zip(*t)

l_b = [9, 24, 101]

l_b.sort()


idx = merge_unique_valued_lists(values, ids, l_b)

for x in idx:
    print(l_a[x])

"""c"""

###############################################################################




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

w = 90

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

w = 90

l = get_chunks(df, 100000, 'ID')


l = get_chunks(df, 100000, 'ID')

idx = 0
for x in l:
    start = x[0]
    end   = x[1]

    print("df[" + str(start) + ":" + str(end) + "]")

    df_p = df[start:end]

    q = create_json(df_p)
    filename = "data_all_5MAR2018" + str(idx) + ".json"
    write_json(filename, q)

    idx = idx + 1

w = 90

###########################################
#
# Start from disk

df_syk = pd.read_csv("SYK")

syk_entry_begin = toDaysSinceEpoch("2014-10-01")
syk_entry_end = toDaysSinceEpoch("2014-10-02")

# How many new in this small range:
df_syk.head()

m = (df_syk.F0 >= syk_entry_begin) & (df_syk.F0 < syk_entry_end)

q = df_syk[m]

d = {}

def note(x):
    d[x] = 1

q.IDX.apply(note)

### Population is d

m = df_syk.IDX.isin(d)

df_syk = df_syk[m]

df_syk.drop(['Unnamed: 0', 'DOCID', 'F1', 'DIAG'], axis = 1, inplace = True)
df_syk.columns = ['F', 'T', 'SAKSKODE', 'ID']


df_aa = pd.read_csv("AA")

m = df_aa.IDX.isin(d)

df_aa = df_aa[m]

df_aa.drop(['Unnamed: 0'], inplace = True, axis = 1)
df_aa.head()
df_aa.columns = ['F', 'T', 'SAKSKODE', 'ID']

df_meld = pd.read_csv("MELDEKORT")

m = df_meld.IDX.isin(d)

df_meld = df_meld[m]

df_meld.drop(['Unnamed: 0'], inplace = True, axis = 1)
df_meld.columns =  ['F', 'T', 'SAKSKODE', 'ID']

df_syk.head()
df_meld.head()
df_aa.head()

df = pd.concat([df_syk, df_aa, df_meld])

df = df.sort_values(by=['ID'])

df = df.set_index('ID')

s = pd.factorize(df.index)

se = pd.Series (s[0])

df.index = se

len (df)

df['ID'] = df.index

df0 = df[:10000]
df1 = df[10000:20000]
df2 = df[20000:30000]
df3 = df[30000:40000]
df4 = df[40000:50000]
df5 = df[50000:60000]
df6 = df[60000:]

df['ID'] = df.index


# Test on small set:

s = df[0:49]

q = create_json(s)


#####################################################################################################
#
# Misc post
#

<<<<<<< HEAD
df_vedtak.drop('PERSON_ID', inplace = True, axis = 1)
df_vedtak.columns= ['ID', 'F', 'T', 'C']

df_birth.drop('T', inplace = True, axis = 1)

df_syk4.drop('FK_PERSON1', inplace = True, axis = 1)
df_syk4['C'] = 'SYKM'  
=======
df['F'] = df['F'].apply (lambda x: (x - pd.datetime(1970, 1, 1)).days)
df['T'] = df['T'].apply (lambda x: (x - pd.datetime(1970, 1, 1)).days) 
>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840

df_fravar.drop('FK_PERSON1', inplace = True, axis = 1)


df_aa.head()
df_aa['IDX'] = df_aa.ID.apply(GetIDX_FROM_FIDK)


<<<<<<< HEAD
df_meldekort['IDX'] = df_meldekort.ID.apply(GetIDX_FROM_FIDK)
df_vedtak['IDX'] = df_vedtak.ID.apply(GetIDX_FROM_FIDK)
df_birth['IDX'] = df_birth.ID.apply(GetIDX_FROM_FIDK)
df_syk4['IDX'] = df_syk4.FID.apply(GetIDX_FROM_FIDK)
df_fravar['IDX'] = df_fravar.FID.apply(GetIDX_FROM_FIDK)

df_fravar.head()

=======
df['C'] = 'AAREG'   #not 'AA' - in use for vedtak.
>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840

#collect all FIDs in set. Create function.
s = set()

df_fravar.FID.apply(lambda x: s.add(x))
df_syk4.FID.apply(lambda x: s.add(x))
df_birth.ID.apply(lambda x: s.add(x))
df_vedtak.ID.apply(lambda x: s.add(x))
df_meldekort.ID.apply(lambda x: s.add(x))
df_aa.ID.apply(lambda x: s.add(x))

l2 = list(s)  #run above on this list

#end function

# create try/catch.
s = create_json_IDX_F_T_SAKSKODE(df_aa)
write_json("datafeb_aa.json", s)

s = create_json_IDX_F_T_SAKSKODE(df_meldekort)
write_json("datafeb_meld.json", s)

s = create_json_IDX_F_T_SAKSKODE(df_vedtak)
write_json("datafeb_vedtak.json", s)

df_birth.head()

df_birth.columns = ['ID', 'SAKSKODE', 'F', 'T', 'IDX']

# Store: F, T, SAKSKODE, IDX, 
s = create_json_IDX_F_T_SAKSKODE(df_birth)
write_json("datafeb_birth.json", s)

df_fravar.head()
df_fravar.columns = ['VID', 'F', 'T', 'FID', 'SAKSKODE', 'IDX']

# Store: F, T, SAKSKODE, IDX, VID

s = create_json_IDX_F_T_SAKSKODE_VIRKID(df_fravar)

write_json("datafeb_fravar.json", s)


df_syk4.columns = ['DOCID', 'F0', 'F1', 'T1', 'DIAG', 'FID', 'SAKSKODE', 'IDX']
s = create_json_IDX_F_F_T_SAKSKODE_DOCID_DIAG_(df_syk4)

write_json("datafeb_syk.json", s)


#Add noised columns and write to csv withouot FID


df_meldekort.dropna(inplace = True)

df_meldekort.head()
df_meldekort.F = addNoise(df_meldekort.F, 0)

df_meldekort.T = addNoise(df_meldekort.T, 3)



df_aa.head()
df_aa.dropna(inplace = True)


df_aa.F = addNoise(df_aa.F, 3)
df_aa.T = addNoise(df_aa.T, 3)

df_fravar.head()

df_fravar.F = addNoise(df_fravar.F, 3)
df_fravar.T = addNoise(df_fravar.T, 3)


df_fravar.dropna(inplace = True)


df_syk4.F0 = addNoise(df_syk4.F0, 3)
df_syk4.F1 = addNoise(df_syk4.F1, 3)
df_syk4.T1 = addNoise(df_syk4.T1, 3)

df_syk4.dropna(inplace = True)

df_syk4.F0 = df_syk4.F0.astype(int)
df_syk4.F1 = df_syk4.F1.astype(int)
df_syk4.T1 = df_syk4.T1.astype(int)

df_syk4.head()

df_vedtak.head()

df_vedtak.F = addNoise(df_vedtak.F, 3)
df_vedtak.T = addNoise(df_vedtak.T, 3)

df_birth.head(9)


## REMOVE ID and write

df_vedtak.head()
df_vedtak.drop(['ID'], axis = 1, inplace = True)
df_vedtak.head()

df_vedtak.to_csv("VEDTAK")

del df_vedtak

df_meldekort.head()
df_meldekort.drop(['ID'], axis = 1, inplace = True)
df_meldekort.to_csv("MELDEKORT")

del df_meldekort

df_syk4.drop(['FID'], axis = 1, inplace = True)

df_syk4.to_csv("SYK")
del df_syk4

df_fravar.head(9)
df_fravar.drop(['FID'], axis = 1, inplace = True)

df_fravar.to_csv("FRAV")


df_birth.drop(['ID'], axis = 1, inplace = True)
df_birth.head()

df_birth.to_csv("AKTIV")

df_aa.drop(['ID'], axis = 1, inplace = True)
df_aa.head()

df_aa.to_csv("AA")

##############################################################################################
#
#   Select all vedtaks
#   With arena id only
#  

df_vedtak.head()

df_vedtak['FODSELSNR'] = df_vedtak['FODSELSNR'].astype(str)

print("Vedtak for population count = " + str (len(df_vedtak)))

#####################################


l_a = df_aa.OFFENTLIG_IDENT.tolist()
l_b = l_fid
i = merge_uniquesorted_lists(l_a, l_b)

# Todo:
# Finds matches, but we need to go back to original indices
# Also: convert string to numbers before sorting.

########################################################################################################




# ---- Read in configuration file and create json dictionary.

# Create class for config. Include 'reload' method.

config_file = ""

f = open(config_file,"r")
config = f.read()
f.close()

config_data = json.loads(config)
print(f"config_data size = {len(config_data)}")



d_all = db_load(False)




=======
<<<<<<< HEAD
# Create randomized birthdays (slow operation: 10 minutes)
s = p.FID.apply(lambda x: get_random_epoch_birth_day(x))

p = p.assign(BIRTH = s)

s = p.FID.apply(lambda x: get_random_epoch_birth_day(x))

p['SAKSKODE'] = 'AKTIV'

p['FRA'] = s + (18 * 365)
p['TIL'] = s + (67 * 365)

p.drop(['BIRTH'], inplace = True, axis = 1)





"""c"""

d_all = db_load(False)

df_syk = d_all['syk']

d_backup = d_all

p = d_all['pmap']

p.columns = ['FID', 'FK', 'A']

id_converter = IDConverter(p)

df_syk = preprocess_syk(d_all['syk'])

df_fravar = preprocess_fravar(d_all['fravar'])


"""c"""

#### l_fid and index dictionary #################

i = np.arange(len (l_fid))
dictFID_TO_IDX = dict(zip (l_fid, i))

i = np.arange(len (l2))
dictFID_TO_IDX = dict(zip (l2, i))

def GetIDX_FROM_FIDK(fid):
    try:
        return dictFID_TO_IDX[fid]
    except:
        return "None"
<<<<<<< HEAD

"""c"""


"""c"""

df_aa = d_all['aa']



"""c"""

<<<<<<< HEAD
    # Get rid of high values. CLIP AT MAX

    max = (future - epoch).days

    df_aa.FRA = df_aa.FRA.clip_upper(max)
    df_aa.TIL = df_aa.TIL.clip_upper(max)

    # Remove invlaid AAREG ranges.

    # m is valid mask: 
    m = (df_aa.FRA < df_aa.TIL)

    d = GetTrueAndFalse(m)

    print(f"Removing Zero or negative AAREG intervals: {d['False']} out of {d['True'] + d['False']}")

    df_aa = df_aa[m]

    df_aa['SAKSKODE'] = 'AAREG'
=======

>>>>>>> 0ece48254089977a6525f0b207088ea052d6e201

    sFRA = GetDaysSinceEpoch(df_aa.FRA, epoch, future)
    sTIL = GetDaysSinceEpoch(df_aa.TIL, epoch, future)

<<<<<<< HEAD
df_aa.head()




#
# Add noise
#

# Todo: Add noise before cleaning.
# Todo: Add noise in one direction from actual value, or allow specification of range.

df_aa.head()
s = addNoise(df_aa.FRA, 3)
t = addNoise(df_aa.TIL, 3)
df_aa = df_aa.assign(FRA = s)
df_aa = df_aa.assign(TIL = t)
=======
    df_aa = df_aa.assign(FRA = sFRA)
    df_aa = df_aa.assign(TIL = sTIL)


    # Get rid of high values. CLIP AT MAX

    max = (future - epoch).days

    df_aa.FRA = df_aa.FRA.clip_upper(max)
    df_aa.TIL = df_aa.TIL.clip_upper(max)

    # Remove invlaid AAREG ranges.

    # m is valid mask: 
    m = (df_aa.FRA < df_aa.TIL)

    d = GetTrueAndFalse(m)
<<<<<<< HEAD

    print(f"Removing Zero or negative AAREG intervals: {d['False']} out of {d['True'] + d['False']}")

    df_aa = df_aa[m]

=======

    print(f"Removing Zero or negative AAREG intervals: {d['False']} out of {d['True'] + d['False']}")

    df_aa = df_aa[m]

>>>>>>> a26b7b55ebf0dd2cda569dfd77d417b9de45ff94
    df_aa['SAKSKODE'] = 'AAREG'

"""c"""

df_aa.head()
<<<<<<< HEAD




### Collect FIDs

s = set()

_ = df_p.FID.apply(s.add)
_ = df_aa.FID.apply(s.add)
_ = df_vedtak.FID.apply(s.add)
_ = df_meldekort.FID.apply(s.add)
_ = df_syk.FID.apply(s.add)
_ = df_fravar.FID.apply(s.add)

lKey = list (s)
nKeys = len (s)

arr = np.arange(nKeys)
np.random.shuffle(arr)

d2 = dict (zip (lKey, arr))

#d2 look up from FID to IDX


def addIDX(df, d):
    s = df.FID.apply(lambda x: d[x])
    return df.assign(IDX = s)
=======
    df_meldekort.columns = ['FID', 'FRA']

    df_meldekort = df_meldekort.assign(TIL = s)

    df_meldekort['SAKSKODE'] = 'MELD'
=======


#
# MELDEKORT
#

preprocess meldekort

df_meldekort = d_all['meldekort']

df_meldekort.head()

df_meldekort.isnull().sum()

df_meldekort.dropna(inplace = True)

MAXDAY = (future - epoch).days

m = df_meldekort[1] >= MAXDAY - 14
>>>>>>> 0ece48254089977a6525f0b207088ea052d6e201

# Remove high values

<<<<<<< HEAD
### Create IDX

df_p = addIDX(df_p, d2)
df_vedtak = addIDX(df_vedtak, d2)
df_aa = addIDX(df_aa, d2)
df_meldekort = addIDX(df_meldekort, d2)
df_syk = addIDX(df_syk, d2)
df_fravar = addIDX(df_fravar, d2)

### DROP FID AND FK

df_p.drop(["FID", "FK", "A"], inplace = True, axis = 1)
df_vedtak.drop(["FID"], inplace = True, axis = 1)
df_aa.drop(["FID"], inplace = True, axis = 1)
df_syk.drop(["FID", "FK"], inplace = True, axis = 1)
df_fravar.drop(["FID", "FK"], inplace = True, axis = 1)
df_meldekort.drop(["FID"], inplace = True, axis = 1)


########################
#
#   get_from_df
#

def get_from_df(df, idx):
    m = (df.IDX == idx)
    return df[m]

"""c"""

idx = 39031  # Rich history
idx = 9011  # Rich
idx = 88    # AA only
idx = 10001

q = {}

q['p'] = get_from_df(df_p, idx)
q['a'] = get_from_df(df_aa, idx)
q['s'] = get_from_df(df_syk, idx)
q['f'] = get_from_df(df_fravar, idx)
q['m'] = get_from_df(df_meldekort, idx)
q['v'] = get_from_df(df_vedtak, idx)
print(f"{idx}: [p:{len(q['p'])}] [a:{len(q['a'])}] [s:{len(q['s'])}] [f:{len(q['f'])}] [m:{len(q['m'])}] [v:{len(q['v'])}]")

# ...Create json struct for one user

# ...Create one data frame for all

# All columns collected:
# IDX, C, F, T, DID, D, F1, VID


df_p.head()
df_aa.head()
df_syk.head()
df_fravar.head()
df_meldekort.head()
df_vedtak.head()
 
# NAN on VID
df_p["VID"] = -1
df_aa["VID"] = -1
df_syk["VID"] = -1
df_meldekort["VID"] = -1
df_vedtak["VID"] = -1

# NA on D
df_p["D"] = "NA"
df_aa["D"] = "NA"
df_fravar["D"] = "NA"
df_meldekort["D"] = "NA"
df_vedtak["D"] = "NA"

# 0 on F1
df_p["F1"] = 0
df_aa["F1"] = 0
df_fravar["F1"] = 0
df_meldekort["F1"] = 0
df_vedtak["F1"] = 0

# -1 on DID
df_p["DID"] = -1
df_aa["DID"] = -1
df_fravar["DID"] = -1
df_meldekort["DID"] = -1
df_vedtak["DID"] = -1

# Rearrange columns
l = ['IDX', 'C', 'F', 'T', 'DID', 'D', 'F1', 'VID']

df_p = df_p[l]
df_aa = df_aa[l]
df_fravar = df_fravar[l]
df_syk = df_syk[l]
df_meldekort = df_meldekort[l]
df_vedtak = df_vedtak[l]

b = pd.concat([df_p, df_aa, df_fravar, df_syk, df_meldekort, df_vedtak], ignore_index=True)

# Slow ( 5 minutes)
b_sorted = b.sort_values(by=['IDX'])

# Slow ( 2 minutes)
b_sorted["C"].value_counts()

# 13.17 - 13.24
b_sorted.to_pickle("14mar2018.pkl")

df = pd.read_pickle("14mar2018.pkl")

#DID -> shuffled sequence.
#Todo: Do on syk above.

b_sorted = b_sorted.reset_index(drop=True)

s = set()

_ = b_sorted.DID.apply(lambda x: s.add(x))

lKey = list (s)
nKeys = len (s)

arr = np.arange(nKeys)
np.random.shuffle(arr)

d2 = dict (zip (lKey, arr))

s = b_sorted.DID.apply(lambda x: d2[x])

b_sorted = b_sorted.assign(DID = s)

#VID -> shuffle
#Todo: Fix on fravar

s = set()

_ = b_sorted.VID.apply(lambda x: s.add(x))

lKey = list (s)
nKeys = len (s)

arr = np.arange(nKeys)
np.random.shuffle(arr)

d2 = dict (zip (lKey, arr))

s = b_sorted.VID.apply(lambda x: d2[x])

b_sorted = b_sorted.assign(VID = s)

#
#
# https://helsedirektoratet.no/Lists/Publikasjoner/Attachments/1028/Diagnosestatistikk%20for%20kommunale%20helse-%20og%20omsorgstjenester.%20Data%20fra%20IPLOS-registeret.%20IS-0511.pdf
# https://ehelse.no/Lists/Publikasjoner/Attachments/9/ICPC-2-bok_110304.pdf
#

b_sorted.C = b_sorted.C.astype('category')
b_sorted.D = b_sorted.D.astype('category')

mem_info_MB(b_sorted)

# 13709.58431
b_sorted.to_pickle("14mar2018_V3.pkl")

# Todo cap or remove F (min) early.
b_sorted.F.min()
=> -717874
=======
df_meldekort = df_meldekort[~m]

s = df_meldekort[1] + 14

df_meldekort.columns = ['FID', 'FRA']

df_meldekort = df_meldekort.assign(TIL = s)

df_meldekort['SAKSKODE'] = 'MELD'
>>>>>>> a26b7b55ebf0dd2cda569dfd77d417b9de45ff94

#
# VEDTAK
#
<<<<<<< HEAD

df_vedtak = d_all['vedtak']

def preprocess_vedtak(df_vedtak):

    df_vedtak.columns = ["FID", "A_ID", "FRA", "TIL", "SAKSKODE"]

    df_vedtak.isnull().sum()

    m = df_vedtak.FID.isnull()

    m.value_counts()
=======

df_vedtak = d_all['vedtak']

df_vedtak.head()

df_vedtak.columns = ["FID", "A_ID", "FRA", "TIL", "SAKSKODE"]

df_vedtak.isnull().sum()

m = df_vedtak.FID.isnull()

m.value_counts()

# Drop where FID is null

df_vedtak = df_vedtak[~m]

df_vedtak.head()

# Drop Arena ID
df_vedtak.drop(['A_ID'], inplace = True, axis = 1)

m = df_vedtak.FRA > df_vedtak.TIL

#WARNING: MANY (8 %)
GetTrueAndFalse(m)

df_vedtak = df_vedtak[~m]

# CLIP TO MAXDAY

df_vedtak.FRA = df_vedtak.FRA.clip_upper(MAXDAY)
df_vedtak.TIL = df_vedtak.TIL.clip_upper(MAXDAY)

MINDAY = (pd.datetime(1980, 1, 1) - epoch).days

df_vedtak.FRA = df_vedtak.FRA.clip_lower(MINDAY)
df_vedtak.TIL = df_vedtak.TIL.clip_lower(MINDAY)
>>>>>>> a26b7b55ebf0dd2cda569dfd77d417b9de45ff94

    # Drop where FID is null

    df_vedtak = df_vedtak[~m]

    # Drop Arena ID
    df_vedtak.drop(['A_ID'], inplace = True, axis = 1)

    m = df_vedtak.FRA > df_vedtak.TIL

    #WARNING: MANY (8 %)
    GetTrueAndFalse(m)

    df_vedtak = df_vedtak[~m]

    # CLIP TO MAXDAY

    df_vedtak.FRA = df_vedtak.FRA.clip_upper(MAXDAY)
    df_vedtak.TIL = df_vedtak.TIL.clip_upper(MAXDAY)

    MINDAY = (pd.datetime(1980, 1, 1) - epoch).days































