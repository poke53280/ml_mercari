

import numpy as np
import pandas as pd

DATA_DIR = "C:\\NORDBANK\\"

df = pd.read_csv(DATA_DIR + 'wiki.no.vec', chunksize = 1000, delimiter = ' ')


df_c = pd.DataFrame(df.get_chunk(1500))

df_c = df_c.reset_index()

df_c
