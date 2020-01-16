
import dask
import time


def funcA(x, y):
    print(x, y)
    time.sleep(10)
    return x + 3


def funcB(x, y):
    print(x, y)
    time.sleep(5)
    return x + 9


l_res = []

l_res.append(dask.delayed(funcA)(9, 17))
l_res.append(dask.delayed(funcB)(11, 11))

tot_res = dask.compute(*l_res)


#     df = df.assign(**dict(zip (l_categorical, tot_res)))












