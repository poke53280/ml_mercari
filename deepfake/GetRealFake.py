

import numpy as np
from pathlib import Path
from mp4_frames import get_output_dir
from mp4_frames import get_ready_data_dir
import matplotlib.pyplot as plt



output_dir = get_output_dir()
ready_dir = get_ready_data_dir()


l_files = list (sorted(output_dir.iterdir()))

l_files = [x for x in l_files if "npy" in x.suffix]

iFile = 0

photos = list()
labels = list()

for x in l_files:

    # print (x)
    anData = np.load(x)

    video_size = 32

    W = 256
    H = 1

    anData = anData.reshape(-1, video_size, W, 3)

    anReal = anData[:7]
    anFake = anData[7:14]

    for i in range(7):
        photos.append(anReal[i])
        labels.append(0.0)
        photos.append(anFake[i])
        labels.append(1.0)

    isLast = (x == l_files[-1])
    
    if isLast or len(photos) > 1000:
        photos = np.asarray(photos)
        labels = np.asarray(labels)
        photos = photos/255.0

        filepath_photo = ready_dir / f"photos_{iFile:04}.npy"
        filepath_label = ready_dir / f"labels_{iFile:04}.npy"

        np.save(filepath_photo, photos)
        np.save(filepath_label, labels)

        iFile = iFile + 1
        photos = list()
        labels = list()



