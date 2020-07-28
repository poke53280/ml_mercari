


from mp4_frames import get_ready_data_dir
from PIL import Image
from line_sampler import get_line

from VideoManagerImpl import VideoManager

import numpy as np
import pandas as pd
from multiprocessing import Pool


####################################################################################
#
#   line_sampler
#

def line_sampler(iCluster):


    input_dir = get_ready_data_dir() / f"c2_{iCluster}"

    if input_dir.is_dir():
        pass
    else:
        print(f"No data for cluster {iCluster}")
        return


    l_files = list (sorted(input_dir.iterdir()))


    file_tuples = set()

    for x in l_files:
        l_x = x.stem.split("_")

        file_tuple = (l_x[0], l_x[1])
        file_tuples.add(file_tuple)


    l_angle = []
    l_pix   = []
    l_test  = []
    l_real  = []
    l_file  = []
    l_seq   = []
    l_error = []

    for x in list(file_tuples):
        mask_path = input_dir / f"{x[0]}_{x[1]}_m.png"
        test_path = input_dir / f"{x[0]}_{x[1]}_t.png"
        real_path = input_dir / f"{x[0]}_{x[1]}_r.png"

        isTrioValid = mask_path.is_file() and test_path.is_file() and real_path.is_file()

        if not isTrioValid:
            continue

        try:
            mask_image = np.asarray(Image.open(mask_path))
            test_image = np.asarray(Image.open(test_path))
            real_image = np.asarray(Image.open(real_path))
        except Exception as err:
            continue

        isSameSize = (test_image.shape == real_image.shape) and (mask_image.shape == (real_image.shape[0],real_image.shape[1]) )

        if not isSameSize:
            continue

        width = test_image.shape[1]

        if width < 50:
            continue

        num_samples_per_pair = 3000

        for i in range(num_samples_per_pair):

            rAngle, rPix, rErrorRate, test_line, real_line = sample(mask_image, test_image, real_image, 50)

            l_angle.append(rAngle)
            l_pix.append(rPix)
            l_test.append(test_line)
            l_real.append(real_line)
            l_file.append(x[0])
            l_seq.append(x[1])
            l_error.append(rErrorRate)

    df = pd.DataFrame({'file':l_file, 'seq':l_seq, 'pix':l_pix, 'angle':l_angle, 'error': l_error, 'real':l_real, 'test':l_test})

    df.to_pickle(get_ready_data_dir() / f"cline_{iCluster}.pkl")


####################################################################################
#
#   sample
#
    
def sample(mask_image, test_image, real_image, n_target_size):
    width = test_image.shape[1]
    height = test_image.shape[0]

    h0 = np.random.choice(height)
    h1 = np.random.choice(height)

    p0 = (0,h0)
    p1 = (width-1, h1)

    p0 = (*p0, 0)
    p1 = (*p1, 0)

    l = get_line(np.array(p0), np.array(p1))
    
    l = l[0:2]

    l = l.astype(np.int32)
    

    n_border = int ((l.shape[1] - n_target_size) / 2)

    assert n_border >= 0

    l = l[:, n_border:n_border + n_target_size]

    assert l.shape[1] == n_target_size

    test_line = test_image[l[1], l[0]]
    mask_line = mask_image[l[1], l[0]]
    real_line = real_image[l[1], l[0]]

    rErrorRate = mask_line.sum() / mask_line.shape[0]

    rAngle = np.arctan2( (h1 - h0), (width - 1))
    rPix = np.sqrt(width * width + height * height)

    return rAngle, rPix, rErrorRate, test_line, real_line


####################################################################################
#
#   __main__
#

if __name__ == '__main__':
    
    v = VideoManager()

    df = v._df

    l_tasks = list(np.unique(df.cluster))

    num_threads = 20

    print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(line_sampler, l_tasks)




