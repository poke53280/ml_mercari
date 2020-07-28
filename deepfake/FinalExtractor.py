

import numpy as np
from mp4_frames import read_video

from face_detector import MTCNNDetector

from pathlib import Path

from VideoManagerImpl import VideoManager

from featureline import find_spaced_out_faces_boxes
from featureline import get_random_face_box_from_z
from featureline import _get_face_boxes


from multiprocessing import Pool
import cv2
from mp4_frames import get_ready_data_dir
from line_sampler import get_line

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale






def sample_video_set(mtcnn_detector, entry):

    outputsize = 128 + 64
    n_target_size = 100
    num_fake_samples_per_frame = 500

    l_line = []
    l_fake = []
    l_angle = []
    l_pix = []

    orig_path = entry[0]

    try:
        orig_video = read_video(orig_path, 0)
    except Exception as err:
        print(err)
        return

    z_max = orig_video.shape[0]
    y_max = orig_video.shape[1]
    x_max = orig_video.shape[2]

    l_all = entry[1]

    for test_path in l_all:

        print ("     " + str(test_path))

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

        frame_min = np.array(list(d_faces.keys())).min()
        frame_max = np.array(list(d_faces.keys())).max()

        if frame_max == frame_min:
            print("No faces found")
            continue

        for i in range(50):

            z_sample = np.random.choice(range(frame_min, frame_max))

            bb_min, bb_max = get_random_face_box_from_z(d_faces, z_sample, x_max, y_max, z_max)

            real_image, test_image = cut_frame(bb_min, bb_max, orig_video, test_video, z_sample, outputsize, False)
                
            for i in range(num_fake_samples_per_frame):
                rAngle, rPix, test_line = sample(test_image, n_target_size)

                l_line.append(test_line)
                l_fake.append(True)
                l_angle.append(rAngle)
                l_pix.append(rPix)                                


            for i in range (num_fake_samples_per_frame):
                Angle, rPix, test_line = sample(real_image, n_target_size)

                l_line.append(test_line)
                l_fake.append(False)
                l_angle.append(rAngle)
                l_pix.append(rPix) 

    df = pd.DataFrame({'fake': l_fake, 'angle': l_angle, 'pix': l_pix, 'data': l_line})
    
    return df


####################################################################################
#
#   get_linear_prediction_channel
#

def get_linear_prediction_channel(t, x, isDraw):

    # transforming the data to include another axis
    t = t[:, np.newaxis]
    x = x[:, np.newaxis]

    polynomial_features= PolynomialFeatures(degree=12)
    t_poly = polynomial_features.fit_transform(t)


    model = LinearRegression()
    model.fit(t_poly, x)

    x_pred = model.predict(t_poly)

    if isDraw:
        mse = mean_squared_error(x, x_pred)
        print(f"MSE: {mse}")

        plt.scatter(t, x, s=10)
        # sort the values of x before line plot
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(t,x_pred), key=sort_axis)
        t_out, x_poly_pred = zip(*sorted_zip)
        plt.plot(t_out, x_poly_pred, color='m')
        plt.show()

    return x_pred


####################################################################################
#
#   get_linear_prediction_row
#

def get_linear_prediction_row(t, data0, isDraw):
    px = get_linear_prediction_channel(t, data0[:, 0], isDraw).reshape(-1)
    py = get_linear_prediction_channel(t, data0[:, 1], isDraw).reshape(-1)
    pz = get_linear_prediction_channel(t, data0[:, 2], isDraw).reshape(-1)

    data0_ = np.array([px, py, pz]).T
    return data0_




####################################################################################
#
#   process_part
#


def process_part(iCluster):

    print(f"process_part {iCluster} starting...")

    assert get_ready_data_dir().is_dir()

    output_dir = get_ready_data_dir()

    v = VideoManager()

    l_d = v.get_cluster_metadata(iCluster)

    mtcnn_detector = MTCNNDetector()

    for entry in l_d:
        orig_path = entry[0]

        file_base = output_dir / f"c_{iCluster}_{orig_path.stem}"

        filename_df = file_base.with_suffix(".pkl")
        filename_np = file_base.with_suffix(".npy")

        isJobDone = filename_df.is_file() and filename_np.is_file()

        if isJobDone:
            continue


        print (str(orig_path))

        df = sample_video_set(mtcnn_detector, entry)
        print(f"Saving {str(file_base)}...")
        df.to_pickle(filename_df)

        print(f"Preprocessing {str(file_base)}...")

        data = np.stack(df.data.values)
        data = preprocess_input(data)
        np.random.shuffle(data)

        np.save(filename_np, data)

        print(f"Videoset {str(file_base)} done.")


    print(f"Cluster {iCluster} done.")

####################################################################################
#
#   __main__
#

if __name__ == '__main__':
    
    v = VideoManager()

    df = v._df

    l_tasks = list(np.unique(df.cluster))

#   process_part(4)

    num_threads = 20

    print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(process_part, l_tasks)

