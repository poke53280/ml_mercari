
#
# Returns series with value randomly changed +/- input number of days
#

import numpy as np
import pandas as pd
import cx_Oracle
import json

#########################################################################
#
# DataProvider
#
# Data input interface
#

class DataProvider:

    def __init__(self, configtext):
        self._config_data = json.loads(configtext)
        print(f"DataProvider config_data size = {len(self._config_data)}")

    def mem_info_MB(self, df):
        mem = df.memory_usage(index=True).sum()
        return mem/ 1000 / 1000

    """c"""

    def report_bandwidth(self, start_time, end_time, df):
    
        processing_time = end_time - start_time
        memMB = self.mem_info_MB (df)
        rBandwidth = memMB / processing_time

        print(f"[Time: {processing_time:.2f}s] [Data: {memMB:.1f} MB] [Bandwidth: {rBandwidth:.2f} MB/s]")

    """c"""

    def getConnection(self, key):
        db_conf = self._config_data[key]

        con_string      = db_conf['connection']
        con_user        = db_conf['user']
        con_password    = db_conf['password']

        c = cx_Oracle.connect(user=con_user, password=con_password, dsn=con_string)

        print(key + ": " + c.version)
        return c

    """c"""

    def getQuery(self, key, queryKey):
        query_string = ""
        try:
            db_conf = self._config_data[key]
            query_string = db_conf[queryKey]

        except KeyError:
            print ("Key(s) not found : '" + key + "' -> '" + queryKey + "'")
            pass

        assert len(query_string) > 0

        if type(query_string) is list:
            query_string = " ".join(query_string)

        return query_string

    ##########################################################################
    #
    #   read_df
    #

    def read_df (self, db_name, db_query):
        conn = self.getConnection(db_name)
        sql = self.getQuery(db_name, db_query)
    
        start_time = time.time()

        cur = conn.cursor()

        cur.arraysize = 100000

        my_exec = cur.execute(sql)

        results = my_exec.fetchall()

        my_exec.close()

        end_time = time.time()

        conn.close()

        df = pd.DataFrame(results)

        self.report_bandwidth(start_time, end_time, df)
        return df

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

def apply_FID_COL(df, GetFID):
    start_time = time.time()
    s = df.FK_PERSON1.apply(GetFID)
    print('[{}] Done FK_FID conversion'.format(time.time() - start_time))

    df = df.assign(FID=s.values)

    df_no_FID = df[(df.FID == "None")]

    nPMissing = 100 * len (df_no_FID)/ len (df_syk)

    print(f"Missing FID on {nPMissing:.2f}% - removing...")

    df2 = df[(df.FID != "None")]

    return df2

"""c"""


