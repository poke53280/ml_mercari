

import pandas as pd
from mp4_frames import get_ready_data_dir



####################################################################################
#
#   load_clives
#

def load_clives(l_load_files):

    l_df_chunk = []
    for x in l_load_files:
        df_chunk = pd.read_pickle(x)
        l_df_chunk.append(df_chunk)


    df = pd.concat(l_df_chunk, axis = 0, ignore_index=True)
    return df


####################################################################################
#
#   get_clive_files
#

def get_clive_files():

    input_dir = get_ready_data_dir()
    l_files = list (sorted(input_dir.iterdir()))
    l_files = [x for x in l_files if "cline" in x.stem]

    return l_files




