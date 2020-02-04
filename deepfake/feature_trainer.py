
import numpy as np
from mp4_frames import get_output_dir
from mp4_frames import get_ready_data_dir
from featureline import get_feature_converter

input_dir = get_output_dir()
assert input_dir.is_dir()

output_dir = get_ready_data_dir()
assert output_dir.is_dir()

d_f = get_feature_converter()

l_files = list (input_dir.iterdir())
l_files = [x for x in l_files if x.suffix == '.npy']
   
l_train_parts = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21]

l_data_train = []
 

# CONTINUE - PERHAPS FEATURE INSIDE FILE LOOP.

for zFeature in list (d_f.keys()):

    iF = d_f[zFeature]
for x in l_files:

    l_x = str(x.stem).split("_")

    if not len (l_x) == 4:
        # Not correct file name format.
        continue

    iPart = int (l_x[1])
    original = l_x[2]
    fake = l_x[3]

    isCollect = (iPart in l_train_parts)

    if not isCollect:
        continue

    data = np.load(x)

    anFeature = data[:, 0]

    m_correct_feature = (anFeature == iF)

    data = data[:, 1:]

    data = data.reshape(-1, 32, 3)

    data = data[m_correct_feature]

    l_data_train.append(data)


anDataTrain = np.concatenate(l_data_train)

# real/fake same sampled line, from all vidoes in same part set, same feature set.

np.save(input_dir / f"train_{zFeature}.npy", anDataTrain)





