

import numpy as np
import pandas as pd
import time
import dask.delayed

from mp4_frames import get_cache_dir



####################################################################################
#
#   read_cache()
#

def read_cache(name):

    cache_dir = get_cache_dir()

    file_path = cache_dir / f"{name}.pkl"
    if file_path.is_file():
        return pd.read_pickle(file_path)
    else:
        return None


####################################################################################
#
#   write_cache()
#

def write_cache(df, name):

    cache_dir = get_cache_dir()

    file_path = cache_dir / f"{name}.pkl"

    df.to_pickle(file_path)



####################################################################################
#
#   create_dataframe0()
#

def create_dataframe0():

    df_cache = read_cache("dataframe0")

    if df_cache is not None:
        print ("create_dataframe0 cached")
        return df_cache

    print ("create_dataframe0 begin")

    aID = np.arange(1000)
    bData = np.random.choice(range(1000), 1000)

    df = pd.DataFrame({'a' : aID, 'b':bData})

    time.sleep(20)

    write_cache(df, "dataframe0")

    print ("create_dataframe0 end")

    return df


####################################################################################
#
#   create_dataframe1()
#

def create_dataframe1():

    df_cache = read_cache("dataframe1")

    if df_cache is not None:
        print ("create_dataframe1 cached")
        return df_cache

    print ("create_dataframe1 begin")


    aID = np.arange(1000)
    bData = np.random.choice(range(1000), 1000)

    df = pd.DataFrame({'c' : aID, 'd':bData})

    time.sleep(20)

    write_cache(df, "dataframe1")

    print ("create_dataframe1 end")

    return df


####################################################################################
#
#   create_dataframe2()
#

def create_dataframe2():

    df_cache = read_cache("dataframe2")

    if df_cache is not None:
        print ("create_dataframe2 cached")
        return df_cache

    print ("create_dataframe2 begin")

    aID = np.arange(1000)
    bData = np.random.choice(range(1000), 1000)

    df = pd.DataFrame({'e' : aID, 'f':bData})

    time.sleep(20)

    write_cache(df, "dataframe2")

    print ("create_dataframe2 end")

    return df


####################################################################################
#
#   _merge()
#

def _merge(x, y, z):
    print ("_merge begin")
    df = pd.concat([x, y, z], axis = 1)
    time.sleep(20)

    print ("_merge end")

    return df


####################################################################################
#
#   merge()
#

def merge():


    df_cache = read_cache("merge")

    if df_cache is not None:
        print ("merge cached")
        return df_cache

    x = dask.delayed(create_dataframe0)()
    y = dask.delayed(create_dataframe1)()   
    z = dask.delayed(create_dataframe2)()

    w = dask.delayed(_merge) (x, y, z)

    df = w.compute()

    write_cache(df, "merge")
   

    return df



merge()