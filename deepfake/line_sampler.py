



from mp4_frames import get_output_dir

from image_grid import get_grid3D
from image_grid import get_bb_from_centers_3D

import numpy as np
import pandas as pd


dir = get_output_dir()

assert dir.is_dir()

l_df = []

for x in dir.iterdir():
    if x.suffix == '.pkl':
        df_original = pd.read_pickle(x)
        l_df.append(df_original)


df = pd.concat(l_df, axis = 0)


# ...

