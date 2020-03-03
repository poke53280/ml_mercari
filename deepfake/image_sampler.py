

import numpy as np
from mp4_frames import read_video

from mp4_frames import get_part_dir
from mp4_frames import get_output_dir

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

def process_part(iPart):

    isDraw = False

    assert get_ready_data_dir().is_dir()

    outputsize = 128

    mtcnn_detector = MTCNNDetector()

    W = 256
    H = 1

    part_dir = get_part_dir(iPart)

    l_d = read_metadata(iPart)

    iSample = 0

    for entry in l_d:

        orig_path = entry[0]

        orig_video = read_video(part_dir / orig_path, 0)

        z_max = orig_video.shape[0]
        y_max = orig_video.shape[1]
        x_max = orig_video.shape[2]

        d_faces = find_spaced_out_faces_boxes(mtcnn_detector, orig_video, 12)

        for zSample in range(16):

            z_sample = np.random.choice(range(0, z_max))

            bb_min, bb_max = get_random_face_box_from_z(d_faces, z_sample, x_max, y_max, z_max)

            rDiagnonal = (bb_max - bb_min) * (bb_max - bb_min)

            n_characteristic_face_size = np.sqrt(rDiagnonal[0] + rDiagnonal[1]).astype(np.int32)

            image_real = orig_video[z_sample].copy()

            x_max = image_real.shape[1]
            y_max = image_real.shape[0]

            sample_size = np.max([int(1.1 * outputsize), 1.1 * n_characteristic_face_size])
            half_size = int (sample_size/2)

            center = 0.5 * (bb_min + bb_max)
            center = center.astype(np.int32)

            center_adjusted = np.array([adjust_box_1d(center[0], half_size, x_max), adjust_box_1d(center[1], half_size, y_max)])

            s_min = center_adjusted - half_size
            s_max = center_adjusted + half_size

            if isDraw:
                image_real = cv2.rectangle(image_real, (s_min[0], s_min[1]), (s_max[0], s_max[1]), (255,0,0), 5)
                plt.imshow(image_real)
                plt.show()

            # Sample frame

            l_test_path = entry[1]
            # l_test_path.append(orig_path)

            for path in l_test_path:

                test_video = read_video(part_dir / path, 0)

                for x in range(12):

                    filename = f"{orig_path[:-4]}_{path[:-4]}_{iSample}"

                    offset_min = s_min
                    offset_max = s_max - outputsize

                    offset_range = offset_max - offset_min

                    offset_lo = np.array([offset_min[0] + np.random.choice(offset_range[0]), offset_min[1] + np.random.choice(offset_range[1])])

                    offste_hi = offset_lo + outputsize

                    sample_frame = np.random.choice(z_max)

                    real_sample = orig_video[sample_frame][offset_lo[1]:offste_hi[1], offset_lo[0]:offste_hi[0]].copy()
                    test_sample = test_video[sample_frame][offset_lo[1]:offste_hi[1], offset_lo[0]:offste_hi[0]].copy()
    
                    image_3 = np.sum((real_sample-test_sample)**2,axis=2)

                    mask = image_3 > 300

                    empty_img = np.zeros((mask.shape[1], mask.shape[0], 3), np.uint8)
                    empty_img[mask] = (255, 0, 0)

                    img_tmp = cv2.cvtColor(empty_img, cv2.COLOR_BGR2RGB)

                    kernel = np.ones((2,2),np.uint8)

                    img_tmp = cv2.dilate(img_tmp,kernel,iterations = 3)

                    img_tmp = cv2.erode(img_tmp,kernel,iterations = 3)

                    erosion = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)


                    im = Image.fromarray(test_sample)
                    im.save(get_ready_data_dir() / (filename +".png"))

                    m = erosion == 0

                    m = m.all(axis = 2)

                    im = Image.fromarray(m)
                    im.save(get_ready_data_dir() / (filename +"_mask.png"))

                    iSample = iSample + 1

                    filename = f"{orig_path[:-4]}_{orig_path[:-4]}_{iSample}"

                    im = Image.fromarray(real_sample)
                    im.save(get_ready_data_dir() / (filename +".png"))

                    m = np.zeros((128, 128), dtype = np.bool)
                    m[:] = True
                    im = Image.fromarray(m)
                    im.save(get_ready_data_dir() / (filename +"_mask.png"))

                    iSample = iSample + 1



####################################################################################
#
#   __main__
#

if __name__ == '__main__':
    outdir_test = get_output_dir()
    assert outdir_test.is_dir()

    file_test = outdir_test / "test_out_cubes.txt"
    nPing = file_test.write_text("ping")
    assert nPing == 4

    l_tasks = list (range(50))

    num_threads = 1

    # print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(process_part, l_tasks)


    
