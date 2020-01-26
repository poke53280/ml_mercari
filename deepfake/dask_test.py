

import dask
import dask.delayed as delayed
import time





def f(video):
    time.sleep(10 * random.random())
    print("done")
    return video.shape[0]


l_dask = []

l_dask.append(delayed(detect_features)(video[9:19]))



res = dask.compute(*l_dask)


f __name__ == '__main__':
    with Pool(2) as p:
        print(p.map(process_part, [0, 1]))





