
import numpy as np
import pandas as pd
import time
import dask.delayed
from cached import cached


####################################################################################
#
#   create_dataframe0()
#

def create_dataframe0(args):
    print (f"create_dataframe0 reports argument a = {args}")
    aID = np.arange(1000)
    bData = np.random.choice(range(1000), 1000)

    time.sleep(20)

    df = pd.DataFrame({'a' : aID, 'b':bData})
    return df


####################################################################################
#
#   create_dataframe1()
#

def create_dataframe1(args):
    aID = np.arange(1000)
    bData = np.random.choice(range(1000), 1000)

    time.sleep(20)

    df = pd.DataFrame({'c' : aID, 'd':bData})

    return df


####################################################################################
#
#   create_dataframe2()
#

def create_dataframe2(args):
    aID = np.arange(1000)
    bData = np.random.choice(range(1000), 1000)

    time.sleep(20)

    df = pd.DataFrame({'e' : aID, 'f':bData})

    return df


####################################################################################
#
#   create_dataframe3()
#

def create_dataframe3(args):
    aID = np.arange(1000)
    bData = np.random.choice(range(1000), 1000)

    time.sleep(20)

    df = pd.DataFrame({'g' : aID, 'h':bData})

    return df


####################################################################################
#
#   create_dataframe4()
#

def create_dataframe4(args):
    aID = np.arange(1000)
    bData = np.random.choice(range(1000), 1000)

    time.sleep(20)

    df = pd.DataFrame({'i' : aID, 'j':bData})

    return df


####################################################################################
#
#   _merge_x_y_z()
#

def _merge_x_y_z(args):
    df = pd.concat([args[0], args[1], args[2]], axis = 1)
    time.sleep(20)
    return df


####################################################################################
#
#   _merge_a_b()
#

def _merge_a_b(args):
    df = pd.concat([args[0], args[1]], axis = 1)
    time.sleep(10)
    return df


####################################################################################
#
#   final_merge()
#

def final_merge(args):
    df = pd.concat([args[0], args[1]], axis = 1)
    time.sleep(10)
    return df


c = Cached(get_cache_dir())


x = dask.delayed(c.cached(create_dataframe0))()
y = dask.delayed(c.cached(create_dataframe1))()   
z = dask.delayed(c.cached(create_dataframe2))()

w = dask.delayed(c.cached(_merge_x_y_z))(x, y, z)

a = dask.delayed(c.cached(create_dataframe3))()
b = dask.delayed(c.cached(create_dataframe4))()

a_b_merge = dask.delayed(c.cached(_merge_a_b))(a, b)

res = dask.delayed(c.cached(final_merge))(w, a_b_merge)

df = res.compute()









