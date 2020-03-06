

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
#   adjust_box_1d
#

def adjust_box_1d(c, half_size, extent):
    b_min = np.max([0, c - half_size])

    c = b_min + half_size

    b_max = np.min([extent, c + half_size])

    b_min = b_max - 2 * half_size

    assert (b_max - b_min) == 2 * half_size
    return (b_max + b_min) // 2


####################################################################################
#
#   process_part
#

def process_part(iCluster):

    isDraw = False

    assert get_ready_data_dir().is_dir()

    output_dir = get_ready_data_dir() / f"c_{iCluster}"

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

            d_faces = find_spaced_out_faces_boxes(mtcnn_detector, test_video, 12)

            for i in range(25):

                z_sample = np.random.choice(range(0, z_max))

                bb_min, bb_max = get_random_face_box_from_z(d_faces, z_sample, x_max, y_max, z_max)

                rDiagnonal = (bb_max - bb_min) * (bb_max - bb_min)

                n_characteristic_face_size = np.sqrt(rDiagnonal[0] + rDiagnonal[1]).astype(np.int32)

                image_real = orig_video[z_sample].copy()
                image_test = test_video[z_sample].copy()

                x_max = image_real.shape[1]
                y_max = image_real.shape[0]

                sample_size = int(1.2 * n_characteristic_face_size)
                half_size = int (sample_size/2)

                center = 0.5 * (bb_min + bb_max)
                center = center.astype(np.int32)

                center_adjusted = np.array([adjust_box_1d(center[0], half_size, x_max), adjust_box_1d(center[1], half_size, y_max)])

                s_min = center_adjusted - half_size
                s_max = center_adjusted + half_size

                real_sample = image_real[s_min[1]:s_max[1], s_min[0]:s_max[0]].copy()
                
                
                test_sample = image_test[s_min[1]:s_max[1], s_min[0]:s_max[0]].copy()
               
                image_3 = np.sum((real_sample-test_sample)**2,axis=2)

                mask = image_3 > 300

                empty_img = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

                empty_img[mask] = (255, 0, 0)

                img_tmp = cv2.cvtColor(empty_img, cv2.COLOR_BGR2RGB)

                kernel = np.ones((2,2),np.uint8)

                img_tmp = cv2.dilate(img_tmp,kernel,iterations = 3)

                img_tmp = cv2.erode(img_tmp,kernel,iterations = 3)

                erosion = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)
                erosion = cv2.resize(erosion, (outputsize, outputsize))

                m = erosion == 0
                m = m.all(axis = 2)


                test_sample = cv2.resize(test_sample, (outputsize, outputsize))


                im_mask = Image.fromarray(m)
                im_test = Image.fromarray(test_sample)

                if isDraw:
                    plt.imshow(im_test)
                    plt.show()

                    plt.imshow(im_mask)
                    plt.show()

                filename = filename_base + f"_{iSample:003}"

                im_test.save(output_dir / (filename + ".png"))
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








