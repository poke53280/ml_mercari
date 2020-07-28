

#File "image_sampler.py", line 209, in process_part
#    bb_min, bb_max = get_random_face_box_from_z(d_faces, z_sample, x_max, y_max, z_max)
#  File "/mnt/disks/tmp_mnt/data/code/featureline.py", line 285, in get_random_face_box_from_z
#    assert z >= anKeys[0]
#
# + Fix assert in opencv
#


import numpy as np
from mp4_frames import read_video

from mp4_frames import get_part_dir
from mp4_frames import get_meta_dir

from mp4_frames import get_ready_data_dir
from mp4_frames import read_metadata
from face_detector import MTCNNDetector

from featureline import find_spaced_out_faces_boxes
from featureline import get_random_face_box_from_z

from face_detector import _fit_1D

import matplotlib.pyplot as plt
import cv2

from easy_face import create_diff_image
from PIL import Image
from multiprocessing import Pool

import argparse
from pathlib import Path

import VideoManager




####################################################################################
#
#   process_part
#

def process_part(iCluster):

    isDraw = False

    assert get_ready_data_dir().is_dir()

    output_dir = get_ready_data_dir() / f"c2_{iCluster}"

    if output_dir.is_dir():
        pass
    else:
        output_dir.mkdir()

    assert output_dir.is_dir()

    v = VideoManager.VideoManager()

    l_d = v.get_cluster_metadata(iCluster)

    outputsize = 128 + 64

    mtcnn_detector = MTCNNDetector()

    orig_path = Path("C:\\Users\\T149900\\Downloads\\dfdc_train_part_07\\dfdc_train_part_7\\crnbqgwbmt.mp4")
    orig_path.is_file()

    test_path = Path("C:\\Users\\T149900\\Downloads\\dfdc_train_part_07\\dfdc_train_part_7\\nwzwoxfcnl.mp4")
    test_path.is_file()



    for entry in l_d:

        orig_path = entry[0]

        print (str(orig_path))

        try:
            orig_video = read_video(orig_path, 0)
        except Exception as err:
            print(err)
            continue

        z_max = orig_video.shape[0]
        y_max = orig_video.shape[1]
        x_max = orig_video.shape[2]


        l_all = entry[1]
        l_all.append(orig_path)
        

        for test_path in l_all:

            print ("     " + str(test_path))

            iSample = 0
            filename_base = f"{test_path.stem}"

            try:
                test_video = read_video(test_path, 0)
            except Exception as err:
                print(err)
                continue

            is_identical_format = (test_video.shape[0] == z_max) and (test_video.shape[1] == y_max) and (test_video.shape[2] == x_max)

            if not is_identical_format:
                print("Not identical formats")
                continue

            d_faces = find_spaced_out_faces_boxes(mtcnn_detector, test_video, 30)

            for i in range(10):

                z_sample = np.random.choice(range(0, z_max))

                bb_min, bb_max = get_random_face_box_from_z(d_faces, z_sample, x_max, y_max, z_max)

                im_mask, im_real, im_test = cut_frame(bb_min, bb_max, orig_video, test_video, z_sample, -1, False)
                
                filename = filename_base + f"_{iSample:003}"
                im_test.save(output_dir / (filename + "_t.png"))
                im_real.save(output_dir / (filename + "_r.png"))
                im_mask.save(output_dir / (filename + "_m.png"))
                iSample = iSample + 1


            


####################################################################################
#
#   __main__
#

if __name__ == '__main__':
    
    v = VideoManager.VideoManager()

    df = v._df

    l_tasks = list(np.unique(df.cluster))

    num_threads = 20

    print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(process_part, l_tasks)








