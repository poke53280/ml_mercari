


from mp4_frames import read_metadata
from mp4_frames import get_part_dir
from mp4_frames import get_output_dir
from mp4_frames import read_video

import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool

from image_grid import get_grid3D_overlap
from image_grid import get_bb_from_centers_3D

import numpy as np
import pandas as pd



####################################################################################
#
#   dataframe_exists
#

def dataframe_exists(iPart, x_real):
    output_dir = get_output_dir()

    x_real[:-4]

    zFilename = output_dir / f"p_{iPart}_{x_real[:-4]}_.pkl" 

    return zFilename.is_file()


####################################################################################
#
#   create_diff_video
#

def create_diff_video(video_real, video_fake, outfile):


    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

    dim = (video_real.shape[2], video_real.shape[1])

    video_tracked = cv2.VideoWriter(str(outfile), fourcc, 25.0, dim)


    for iFrame in range(video_real.shape[0]):
        image_1 = cv2.cvtColor(video_real[iFrame], cv2.COLOR_BGR2RGB)
        image_2 = cv2.cvtColor(video_fake[iFrame], cv2.COLOR_BGR2RGB)


        image_3 = np.sum((image_1-image_2)**2,axis=2)

        cm = plt.get_cmap('viridis')

        colored_image = cm(image_3)

        colored_image = colored_image[:, :, :3]

        colored_image = colored_image * 256
        colored_image = colored_image.astype(np.uint8)
    
        video_tracked.write(colored_image)

    video_tracked.release()




####################################################################################
#
#   show_diff
#

def show_diff(image_real, image_fake):
    image_1 = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(image_fake, cv2.COLOR_BGR2RGB)

    image_3 = np.sum((image_1-image_2)**2,axis=2)

    imgplot = plt.imshow(image_3)
    plt.show()


####################################################################################
#
#   show_full
#

def show_full(image_real, image_fake):
    fig, axes = plt.subplots(1,3, figsize=(30,10))

    image_1 = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(image_fake, cv2.COLOR_BGR2RGB)

    axes[0].imshow(image_1)
    axes[0].title.set_text(f"{x_real.stem}")

    axes[1].imshow(image_2)
    axes[1].title.set_text(f"{x_fake.stem}")

    image_3 = np.sum((image_1-image_2)**2,axis=2)
    axes[2].imshow(image_3)
    axes[2].title.set_text("Difference(MSE)")

    plt.show()


####################################################################################
#
#   get_sampling_cubes
#

def get_sampling_cubes(video_real, video_fake):

    assert video_real.shape == video_fake.shape

    video_d = np.sum((video_real-video_fake)**2,axis=3)

    l_c = get_grid3D_overlap(video_d.shape[0], video_d.shape[1], video_d.shape[2], 32, 1.3)

    l_bb = get_bb_from_centers_3D(l_c, 32)

    l_mean = []

    for bb in l_bb:
        im_min_x = bb[0]
        im_max_x = bb[1]
        im_min_y = bb[2]
        im_max_y = bb[3]
        im_min_z = bb[4]
        im_max_z = bb[5]

        l_mean.append(np.mean(video_d[im_min_x:im_max_x, im_min_y: im_max_y, im_min_z: im_max_z]))


    df = pd.DataFrame({'c': l_c, 'mse' : l_mean})

    x = df.c.map(lambda x: x[0])
    y = df.c.map(lambda x: x[1])
    z = df.c.map(lambda x: x[2])

    df = df.assign(x = x, y = y, z = z)

    df = df.drop('c', axis = 1)

    return df


####################################################################################
#
#   get_sampling_cubes_for_part
#

def get_sampling_cubes_for_part(iPart, output_dir):
    
    l_d = read_metadata(iPart)
    dir = get_part_dir(iPart)

    num_videos = len(l_d)

    print(f"p_{iPart}: Fake detection on part {iPart}. {len(l_d)} original video(s).")

    for idx_key in range(num_videos):

        current = l_d[idx_key]

        x_real = current[0]

        isCompleted = dataframe_exists(iPart, x_real)

        if isCompleted:
            # print(f"p_{iPart}_{x_real}: Already done.")
            continue
        else:
            print(f"p_{iPart}_{x_real}: Starting. {idx_key +1} of {len(l_d)}")


        x_real = dir / x_real
        assert x_real.is_file(), "Error: Original not found"

        vidcap = cv2.VideoCapture(str(x_real))
        
        video_real = read_video(vidcap)

        vidcap.release()

        num_frames = video_real.shape[0]

        l_fakes = current[1]

        l_df_video = []

        for x_fake in l_fakes:
            x_fake = dir / x_fake

            if not x_fake.is_file():
                print(f"   WARNING: p_{iPart}_{x_real.stem}: Not a file: {x_fake}. Situation handled.")
                continue

            print(f"   p_{iPart}_{x_real.stem}: Processing {str(x_fake.stem)}")

            vidcap = cv2.VideoCapture(str(x_fake))
        
            video_fake = read_video(vidcap)

            vidcap.release()

            df_video = get_sampling_cubes(video_real, video_fake)

            df_video = df_video.assign(fake = str(x_fake.stem))

            l_df_video.append(df_video)

        if len(l_df_video) > 0:
            df_video = pd.concat(l_df_video, axis = 0)
            df_video = df_video.assign(original = str(x_real.stem))
            df_video = df_video.assign(part = iPart)
            
            df_video.to_pickle(output_dir / f"p_{iPart}_{x_real.stem}_.pkl")
            print(f"p_{iPart}_{x_real.stem}: Complete.")

        else:
            print(f"p_{iPart}_{x_real.stem}: WARNING: No fakes found. No sampling cubes produced for video.")

    return []


####################################################################################
#
#   process_chunk
#

def process_chunk(iPart):
    output_dir = get_output_dir()
    assert output_dir.is_dir()

    return get_sampling_cubes_for_part(iPart, output_dir)

####################################################################################
#
#   main
#

if __name__ == '__main__':

    outdir_test = get_output_dir()
    assert outdir_test.is_dir()

    file_test = outdir_test / "test_out.txt"
    nPing = file_test.write_text("ping")
    assert nPing == 4

    l_tasks = list(range(25))

    num_threads = len (l_tasks)

    print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(process_chunk, l_tasks)








