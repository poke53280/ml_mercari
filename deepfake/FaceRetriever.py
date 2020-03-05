

from VideoManager import VideoManager


v = VideoManager()

df = v._df

assert df.shape[0] == np.unique(v._df.file).shape[0]

anCluster = np.unique(df.cluster)


l_m = v.get_cluster_metadata(anCluster[0])

