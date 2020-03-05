
import numpy as np
import pandas as pd
import pathlib
from mp4_frames import get_meta_dir
from mp4_frames import read_metadata
from mp4_frames import get_part_dir


####################################################################################
#
#   VideoManager
#

class VideoManager:
    def __init__(self):
        df_c = pd.read_feather(get_meta_dir() / "face_clusters.feather")

        l_parts = []

        for x in range(50):
            path = get_part_dir(x, False)
            isDir = path.is_dir()
            if isDir:
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


        l_file_tuple = list(zip (df.file, df.part))

        l_exists = []

        for x in l_file_tuple:
            filepath = get_part_dir(x[1]) / x[0]
            l_exists.append(filepath.is_file())


        df = df.assign(exists = l_exists)

        num_files = df.shape[0]
        num_originals = np.unique(df.orig).shape[0]
        num_clusters = np.unique(df.cluster).shape[0]

        print(f"num_files = {num_files}, num_originals = {num_originals}, num_clusters = {num_clusters}")

        self._df = df

    def get_cluster_metadata(self, iCluster):

        m = (self._df.cluster == iCluster) & (self._df.exists)
        df = self._df[m].reset_index(drop = True).copy()

        # Filter out originals not existing on file.
        azOriginal = np.unique(df.orig.values)

        l_files = df.file.values

        azOriginal = [x for x in azOriginal if x in l_files]

        m = df.orig.isin(azOriginal)

        df = df[m].reset_index(drop = True)

        l_out = []

        for x in azOriginal:

            original = df[df.file == x]
            fakes = df[(df.orig == x) & (df.file != x)]

            full_file_orig = original.apply(lambda x: get_part_dir(x['part']) / x['file'], axis = 1).iloc[0]

            fill_file_fakes = list(fakes.apply(lambda x: get_part_dir(x['part']) / x['file'], axis = 1))

            l_out.append((full_file_orig, fill_file_fakes))

        return l_out

#
#f = VideoManager()
#
#l_d = f.get_cluster_metadata(575)
#
#for entry in l_d:
#    orig = entry[0]
#    l_fakes = entry[1]
#



