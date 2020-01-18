

import pathlib
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool

def process(df_file):
    df = pd.read_pickle(df_file)
    print (df.shape)

if __name__ == '__main__':

    id = np.random.choice(100, size = 100000)
    data = np.random.choice(1000, size = 100000)

    df = pd.DataFrame({'id' : id, 'data' : data})

    base_folder = pathlib.Path("C:\\Users\\T149900\\tmp")

    tmp_folder = "taskX_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    folder = base_folder / tmp_folder

    folder.mkdir(parents=True, exist_ok=False)

    l_d = []

    for i in range(50):
        m = df.id % 50 == i
        df_file = folder / f"my_df_{i:02}_50.pkl"
        df[m].to_pickle(df_file)
        l_d.append(df_file)

    p = Pool(50)
    p.map(process, l_d)


    for i in range(50):
        df_file = folder / f"my_df_{i:02}_50.pkl"
        df_file.unlink()

    folder.rmdir()