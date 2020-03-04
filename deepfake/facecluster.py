
import numpy as np
import pandas as pd
import pathlib
from mp4_frames import get_meta_dir
from mp4_frames import read_metadata
from mp4_frames import get_part_dir


class FileManager:
    def __init__(self):
        df_c = pd.read_feather(get_meta_dir() / "face_clusters.feather")

        l_parts = []

        for x in range(50):
            if get_part_dir(x, False).is_dir():
                l_parts.append(x)

        l_orig = []
        l_file = []
        l_part = []

        for iPart in l_parts:
            df_meta = read_metadata(iPart)

            for x in df_meta:
                num_fakes = len (x[1])
                l_orig.extend([x[0]]* (num_fakes + 1))
                l_file.append(x[0])
                l_file.extend(x[1])
                l_part.extend([iPart] * (num_fakes + 1))

        df = pd.DataFrame({'orig': l_orig, 'file': l_file, 'part': l_part})

        df = df.merge(df_c, left_on = 'orig', right_on='video')

        df = df.drop(['video', 'chunk'], axis = 1)

        self._df = df


f = FileManager()

num_files = df.shape[0]
num_originals = np.unique(df.orig).shape[0]
num_clusters = np.unique(df.cluster).shape[0]


iCluster = 38

m = df.cluster == iCluster

sClusterOriginals_in_38 = list(np.unique(df[m].orig))

l = [x for x in sClusterOriginals_in_38 if x in list(df.file)]

len (sClusterOriginals_in_38)
len (l)










