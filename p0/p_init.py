

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

import concurrent
from concurrent.futures import ThreadPoolExecutor


os.environ['NLS_NCHAR_CHARACTERSET']='AL16UTF16'
os.environ['NLS_CHARACTERSET']='WE8ISO8859P15'
os.environ['NLS_LANG']='AMERICAN_AMERICA.WE8ISO8859P15'



# ---- Read in configuration file and create json dictionary.

config_file = "X:\\XXX\\XXX\\db_connection.json"

f = open(config_file,"r")
config = f.read()
f.close()


dp = DataProvider(config)

###########################################################################################


def task(n):
    print(f"Launching task {n[0]}: {n[1]}")
    key = n[2]
    df = dp.read_df(n[0], n[1])
    return (key, df)


executor = ThreadPoolExecutor(max_workers=20)


start_time = time.time()

l_tasks = []

l_tasks.append(executor.submit(task, ("A", "sql_syk3", "syk")))
l_tasks.append(executor.submit(task, ("A", "sql_fravar", "fravar")))
l_tasks.append(executor.submit(task, ("A", "sql_pmap", "pmap")))
l_tasks.append(executor.submit(task, ("B", "sql_vedtak", "vedtak")))
l_tasks.append(executor.submit(task, ("B", "sql_meldekort", "meldekort")))
l_tasks.append(executor.submit(task, ("C", "select_large", "aa")))
    
#3277

d = {}

for future in concurrent.futures.as_completed(l_tasks):
    key, df = future.result()
    d[key] = df


end_time = time.time()
=======
dp = DataProvider(config)

#########################################################################
#
#   db_load
#

def db_load(dp, isAlna):

    d = {}

    start_time = time.time()

    d['syk'] = dp.read_df("A", "sql_syk3")

    d['fravar'] = dp.read_df("A", "sql_fravar")

    d['pmap'] = dp.read_df("A", "sql_pmap")

    d['vedtak'] = dp.read_df("B", "sql_vedtak")

    d['meldekort'] = dp.read_df("B", "sql_meldekort")

    d['aa'] = dp.read_df("C", "select_large")

    if isAlna:
        d['alna_pop'] = dp.read_df("B", "sql_population")
>>>>>>> 9787bf0618b2468d9a7325592da30373dac15089

processing_time = end_time - start_time

print(f"Database async load compete. Time: {processing_time:.2f}s]")




d_Keep123 = db_load(dp, False)

####################### ID CONVERTERS #################################


class IDConverter:
    dictFK_TO_FID =  {}
    dictFID_TO_FK = {}
    dictA_TO_FID = {}
    dictFID_TO_A = {}

    def __init__(self, p):

        # Remove entries with invalid FK
        m = (p.FK != -1)
        p = p[m]

        #Check length of FID string
        q = p.FID.apply(lambda x: len(x))

        m = (q == 11)
        del q

        m.value_counts()
        #All FIDs at len 11

        l_fid = p.FID.tolist()
        l_fk = p.FK.tolist()

        self.dictFK_TO_FID = dict(zip (l_fk, l_fid))
        self.dictFID_TO_FK = dict(zip (l_fid, l_fk))

        s_fid = set(l_fid)
        s_fk  = set(l_fk)

        #More FIDs than FKs.
        print(f"More FIDs than FKs: {(len (s_fid) - len (s_fk))}")

        # Several entries with invalid A.
        m = (p.A >= 0)
        m.value_counts()

        # Remove them
        p = p[m]

        len (p)

        l_fid = p.FID.tolist()
        l_a   = p.A.tolist()

        self.dictA_TO_FID  = dict(zip (l_a, l_fid))
        self.dictFID_TO_A  = dict(zip (l_fid, l_a))

        len (dictA_TO_FID)
        len (dictFID_TO_A)

        print(f"A very few more FIDS than A: {(len (dictFID_TO_A) - len (dictA_TO_FID))}")

    def GetFID_FROM_FK(self, fk_person1):
        try:
            fid = self.dictFK_TO_FID[fk_person1]
            assert (len(fid) == 11)
            return fid
        except:
            return "None"

    def GetFK_FROM_FID(self, fid):
        try:
            return self.dictFID_TO_FK[fid]
        except:
            return "None"


"""c"""



p = d_Keep123['pmap']

p.columns = ['FID', 'FK', 'A']

id_converter = IDConverter(p)

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


#
# Returns -1 if not found
#

def Arena_to_IDX(arena_id):
    isFound_Arena_to_FID = arena_id in dict_ARENA_ID_TO_FID

    if isFound_Arena_to_FID:
        fid_str = dict_ARENA_ID_TO_FID[arena_id]

        if fid_str in dictFID_TO_IDX:
            return dictFID_TO_IDX[fid_str]
    return -1

"""c"""

#### Match sykefravaer and fid ###################

df_syk = d_Keep123['syk']

df_syk.head()

df_syk.columns = ["FK", "DID", "F0", "F1", "T0", "D"]

# Is here a mapping to FID in all cases?

q = df_syk.FK.apply(lambda x: not id_converter.GetFID_FROM_FK(x) == "None")

type (q.value_counts())

# Many FKs don't have FID. Investigate.








df_syk.columns = ["FK_PERSON1", "FRA", "TIL"]
df_syk.head()

df_pmap.head()

"""c"""
    
df_syk = apply_FID_COL(df_syk, GetFID_FROM_FK)

<<<<<<< HEAD
df_syk['SAKSKODE'] = 'SYKM'

df_syk.head()


s = apply_FID_COL(df_syk3, GetFID)

df_syk3['FID'] = s


df_fravar = apply_FID_COL(df_fravar, GetFID_FROM_FK)
=======
"""c"""
    
start_time = time.time()

s = df_syk.FK_PERSON1.apply(GetFID)

print('[{}] Done FK_FID conversion'.format(time.time() - start_time))

df_syk = df_syk.assign(FID=s.values)
>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840

df_fravar['SAKSKODE'] = 'SYKF'

df_fravar.head()


"""c"""

# Population is l_fid/l_fk: All with one or more registered sykefravar.

s = set(l_fid)
len(s) == len(l_fid)  #uniques

df_no_FID = df_syk[(df_syk.FID == "None")]

nPMissing = 100 * len (df_no_FID)/ len (df_syk)

nPMissing = 21.2132131

print(f"Missing FID on {nPMissing:.2f}% . XXX Investigate/Viz.")


df_syk2 = df_syk[(df_syk.FID != "None")]

len (df_syk2) + len (df_no_FID) == len (df_syk)

df_syk = df_syk2


# Population is l_fid/l_fk: All with one or more registered sykefravar.

s = set(l_fid)
len(s) == len(l_fid)  #uniques


###################### Create region population ##############################

# l_fid: List of all persons having one or more vedtak at 0326.



print("Null values in population: " + str(df_alna_population['FODSELSNR'].isnull().sum()))

s = df_population['FODSELSNR']


s.dropna(inplace = True)


print("Null values in population: " + str(df_population['FODSELSNR'].isnull().sum()))


l_fid = df_population.FODSELSNR.tolist() 

print("Population count = " + str (len(l_fid)))




r = range(0, 1000000)
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


##########################################################################
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


"""c"""

# Need days since epoch for 1

#input: l_fid
<<<<<<< HEAD

#output: list of randomized birth days in days since epoch




e = []

for x in l_fid:
    d = get_random_epoch_birth_day(x)
    e.append(d)

"""c"""


df_birth = pd.DataFrame(dict(FID = l_fid, birth_epoch = e))

df_birth['SAKSKODE'] = 'AKTIV'

df_birth['FRA'] = df.birth_epoch + (18 * 365)
df_birth['TTL'] = df.birth_epoch + (67 * 365)

df_birth.drop(['birth_epoch'], inplace = True, axis = 1)
=======

#output: list of randomized birth days in days since epoch




e = []

for x in l_fid:
    d = get_random_epoch_birth_day(x)
    e.append(d)

"""c"""


df = pd.DataFrame(dict(FID = l_fid, birth_epoch = e))

df['SAKSKODE'] = 'AKTIV'

df['FRA'] = df.birth_epoch + (18 * 365)
df['TTL'] = df.birth_epoch + (67 * 365)

df.drop(['birth_epoch'], inplace = True, axis = 1)

df_birth = df

df_birth.columns = ['ID', 'SAKSKODE', 'FRA', 'TIL']
>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840

df_birth = df

df_birth.columns = ['ID', 'SAKSKODE', 'FRA', 'TIL']

df_birth.head()


###########################################################################
#
#   Select all vedtaks
#   With arena id only
#  
#



df_vedtak.head()

df_vedtak['FODSELSNR'] = df_vedtak['FODSELSNR'].astype(str)

print("Vedtak for population count = " + str (len(df_vedtak)))

#############################################################################################################################################
#
#   Meldekort
#



df_meldekort.isnull().sum()

df_meldekort.dropna(inplace = True)

print("All meldekorts " + str(len(df_meldekort)))

df_meldekort.head()

df_meldekort['TIL'] = df_meldekort['D'].map(lambda x: x + 14)

<<<<<<< HEAD

df_meldekort['SAKSKODE'] = 'MELD'
=======
# Very slow operation, needs sort vs sort lookup. Or: Try check against set.
#nf = df[df['FID'].isin(l_fid)]

#df_meldekort = nf

#df = df_meldekort

df_meldekort['TIL'] = df_meldekort['D'].map(lambda x: x + 14)


df_meldekort['SAKSKODE'] = 'MELD'


df_meldekort.columns= ['ID', 'F', 'T', 'SAKSKODE']

>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840

df_meldekort.head()

df_meldekort.columns= ['ID', 'F', 'T', 'SAKSKODE']

<<<<<<< HEAD
df_meldekort.head()

=======
>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840
###################################################################################################################
#
# AA-REG
#



l_a = df_aa.OFFENTLIG_IDENT.tolist()
l_b = l_fid
i = merge_uniquesorted_lists(l_a, l_b)

# Todo:
# Finds matches, but we need to go back to original indices
# Also: convert string to numbers before sorting.

# Tuple - sort on first element?


df_aa.isnull().sum()

# Terminate all ongoings to near future
df.TIL.fillna("2019-01-01", inplace = True)

df.isnull().sum()

#No nans now

s = df_aa[444:445].FRA.values[0]

# Takes forever:
def toDaysSinceEpoch(x):
    epoch = datetime.datetime.strptime('1970-01-01', '%Y-%m-%d')
<<<<<<< HEAD
    diff = -1
=======
>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840
    try:
        d = datetime.datetime.strptime(x, '%Y-%m-%d')
        diff = (d - epoch).days

    except ValueError:
        print("Value error: '" + x + "'")
        return "NA"

    return diff

"""c"""


#XXX Takes forever
df_aa['F'] = df_aa['FRA'].apply(toDaysSinceEpoch)

<<<<<<< HEAD
df_aa['T'] = pd.to_datetime(df_aa.TIL, errors = 'coerce')
df_aa['F'] = pd.to_datetime(df_aa.FRA, errors = 'coerce')

df_aa.isnull().sum()

#Drop nans on nan F
df_aa = df_aa.loc[~df_aa['F'].isnull()]

=======
>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840

#Terminate open T
df_aa['T'].fillna(pd.datetime(2019, 1, 1), inplace = True)

df_aa.isnull().sum()

df_aa.head()

df_aa.drop('FRA', inplace=True, axis = 1)
df_aa.drop('TIL', inplace=True, axis = 1)


# Takes ages
s = df_aa['F'].apply(lambda x: (x - pd.datetime(1970, 1, 1)).days)
t = df_aa['T'].apply(lambda x: (x - pd.datetime(1970, 1, 1)).days)

df_aa['F'] = s
df_aa['T'] = t

df_aa.drop('eF', inplace=True, axis = 1)
df_aa.drop('eR', inplace=True, axis = 1)
df_aa.drop('R', inplace=True, axis = 1)

df_aa.head()

df_aa['SAKSKODE'] = 'AAREG'   #not 'AA' - in use for vedtak.

df_aa.columns= ['ID', 'F', 'T', 'SAKSKODE']

#################################################
#
# Misc post
#

<<<<<<< HEAD
df_vedtak.drop('PERSON_ID', inplace = True, axis = 1)
df_vedtak.columns= ['ID', 'F', 'T', 'SAKSKODE']

df_birth.drop('T', inplace = True, axis = 1)

df_syk4.drop('FK_PERSON1', inplace = True, axis = 1)
df_syk4['SAKSKODE'] = 'SYKM'  
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
df['SAKSKODE'] = 'AAREG'   #not 'AA' - in use for vedtak.
>>>>>>> 923d61f871052933d5a215f104b494c22f3ec840

#collect all FIDs in set. Create function.
s = set()

df_fravar.FID.apply(lambda x: s.add(x))
df_syk4.FID.apply(lambda x: s.add(x))
df_birth.ID.apply(lambda x: s.add(x))
df_vedtak.ID.apply(lambda x: s.add(x))
df_meldekort.ID.apply(lambda x: s.add(x))
df_aa.ID.apply(lambda x: s.add(x))

l2 = list(s)  #run aove on this list

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
df_fravar.columns = ['VIRK_ID', 'F', 'T', 'FID', 'SAKSKODE', 'IDX']

# Store: F, T, SAKSKODE, IDX, VIRK_ID

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


# trim_down:

q = df [900:910]

df_new = df.loc[df.SAKSKODE > 365 * (2014 - 1970)]

df = df_new
# to output

df.columns = ['F', 'SAKSKODE', 'T', 'ID']

df.head()




conn_siamo.close()

w = 90

sql_meldekort = "DELETED";

df_meldekort = pd.read_sql(sql_meldekort, con = conn)

df_alna = df_alna.sort_values(by="'person_id'")


# DATALAB ---------------------------------------------------


conn_datalab = getConnection("DATALAB")


cur.close()

conn_datalab.close()


#
#
#  get_qty_and_unit
#
#

def get_qty_and_unit(s):
    
    print(s)


w = 90


#
#
# --------------------------------------------------------------------

if __name__ == "__main__":
    
    y = count()

    GetAA(0);      

    y2 = count();

    if y2 == 0:
        print("No data");
    else:
        q = create_nda()
        print (q)

        d = pd.DataFrame(q)
        print (d)

        

