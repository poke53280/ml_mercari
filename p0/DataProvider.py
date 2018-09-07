
import json
import pandas as pd
import cx_Oracle
import json

import concurrent
from concurrent.futures import ThreadPoolExecutor
import time

#########################################################################
#
# DataProvider
#
# Data input interface
# see example() usage below.
#
#
# Config file structure:
#
#{
#  "A": {
#    "connection": "(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=XXX)(PORT=nnnn))(CONNECT_DATA=(SERVICE_NAME=YYY)))",
#    "user": "XXX",
#    "password": "YYY",
#    "SF4_A": "select x, y, z from table",
#    "SF4_B": "select a, b, c from table",
#  },
#
#  "B": {
#    See A
#  }
#
#}
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

    def async_load(self, l_queries):

        def task(n):
            print(f"Launching task {n[0]}: {n[1]}")
            key = n[2]
            df = self.read_df(n[0], n[1])
            return (key, df)

        executor = ThreadPoolExecutor(max_workers=20)


        start_time = time.time()

        l_tasks = []

        for query in l_queries:
            l_tasks.append(executor.submit(task, query))
    
        d = {}

        for future in concurrent.futures.as_completed(l_tasks):
            key, df = future.result()
            d[key] = df


        end_time = time.time()

        processing_time = end_time - start_time

        print(f"Database async load compete. Time: {processing_time:.2f}s]")

        return d

"""c"""

import os

def example():

    # Looks up config file name from environment variable

    config_file = os.getenv('DB_DATA')
    assert config_file is not None, "No config file found in environment variable DB_DATA"
    print(f"Using database connection file '{config_file}'")


    # Reads in config file. Expected json format as outlined above.
    f = open(config_file,"r")
    config = f.read()
    f.close()


    dp = DataProvider(config)

    # Each query will result in a panda table in the 'd' dictionary.
    l_queries = []

    l_queries.append( ("A", "SF4_A",      "d") )
    l_queries.append( ("A", "SF4_B",      "a") )

    d = dp.async_load(l_queries)

    d['d']
    d['a']



