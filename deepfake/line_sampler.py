


from mp4_frames import get_output_dir
from mp4_frames import get_part_dir
from mp4_frames import get_video_path_from_stem_and_ipart
from mp4_frames import read_video
from mp4_frames import get_line
from image_grid import _get_bb_from_centers_3D
from image_grid import GetSubVolume3D


import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool




####################################################################################
#
#   load_sample_cubes
#

def load_sample_cubes(original, l_fakes, l_ac, nCubeSize, iPart):

    l_bb = _get_bb_from_centers_3D(l_ac, nCubeSize)

    l_video_file = []
    l_video_file.append(original)
    l_video_file.extend(l_fakes)

    d = nCubeSize // 2

    d_cubes = []

    for x in l_video_file:
        print(f"Creating cubes from {x}...")
        video = read_video_from_stem_and_ipart(x, iPart)

        l_cubes = []

        for bb in l_bb:
            cube = GetSubVolume3D(video, bb)
            assert cube.shape == (nCubeSize, nCubeSize, nCubeSize, 3)

            l_cubes.append(cube)
        
        d_cubes.append(l_cubes)


    """c"""
    return d_cubes



####################################################################################
#
#   rasterize_lines
#

def rasterize_lines(p):
    l_l = []

    for x in p:
        l = get_line(x[::2], x[1::2])
        assert l.shape[1] >= 16, f"Line is short: {l.shape[1]}"

        l = np.swapaxes(l, 0, 1)
        l = l[:16]
        l = l.astype(np.int32)

        l_l.append(l)

    anLines = np.stack(l_l)
    return anLines


####################################################################################
#
#   get_random_trace_lines
#

def get_random_trace_lines(num_samples):

    p = np.random.randint(0, 32, size = (num_samples, 6), dtype = np.int32)

    direction = np.random.choice(3, num_samples)

    zero_start = direction * 2
    full_stop =  zero_start + 1

    p[np.arange(p.shape[0]), zero_start] = 0
    p[np.arange(p.shape[0]), full_stop] = 31


    l_l = []

    for x in p:
        l = get_line(x[::2], x[1::2])
        assert l.shape[1] >= 16, f"Line is short: {l.shape[1]}"

        l = np.swapaxes(l, 0, 1)
        l = l[:16]
        l = l.astype(np.int32)

        l_l.append(l)

    anLines = rasterize_lines(p)

    return anLines


####################################################################################
#
#   sample_cube
#

def sample_cube(r, anLines):

    l_sample = []

    for l in anLines:

        l_x = l[:, 0]
        l_y = l[:, 1]
        l_z = l[:, 2]

        r_sample = r[l_z, l_y, l_x]
        l_sample.append(r_sample)

    anSamples = np.stack(l_sample)
    return anSamples
"""c"""


####################################################################################
#
#   sample_from_part
#

def sample_from_part(iPart):

    nCubeSize = 32

    output_dir = get_output_dir()
    video_dir = get_part_dir(iPart)

    assert output_dir.is_dir()
    assert video_dir.is_dir()


    l_df = []

    for x in output_dir.iterdir():
        if x.suffix == '.pkl' and x.stem.startswith(f"p_{iPart}"):
            df_original = pd.read_pickle(x)
            l_df.append(df_original)


    num_originals = len (l_df)


    # Basic real-fake match sampling


    for df in l_df:

        original = df.original.iloc[0]
        print (f"Original: {original}")

        l_fakes = list (df.fake.value_counts().index)

        l_samples = []

        for fake in l_fakes:

            dense_threshold = np.quantile(np.array(df[df.fake == fake].mse), 0.999)

            df_dense = df[(df.fake == fake) & (df.mse >= dense_threshold)]

            df_dense = df_dense.sort_values(by = 'mse', ascending = False).reset_index(drop = True)

            ac_x = np.array(df_dense.x)[:10]
            ac_y = np.array(df_dense.y)[:10]
            ac_z = np.array(df_dense.z)[:10]

            l_ac = list(zip(ac_x, ac_y, ac_z))

            d = load_sample_cubes(original, [fake], l_ac, nCubeSize, iPart)

            num_cubes = len (d[0])

            for iCube in range(num_cubes):

                cube_real = d[0][iCube]
                cube_fake = d[1][iCube]

                num_samples = 2000

                anLines = get_random_trace_lines(num_samples)

                real_samples = sample_cube(cube_real, anLines)
                fake_samples = sample_cube(cube_fake, anLines)

                combined_samples = np.hstack([real_samples, fake_samples])

                l_samples.append(combined_samples)
        """c"""
    
        anSamples = np.concatenate(l_samples)

        file_out = output_dir / f"lines_p_{iPart}_{original}.npy"

        np.save(file_out, anSamples)

    
    """c"""




if __name__ == '__main__':
    outdir_test = get_output_dir()
    assert outdir_test.is_dir()

    file_test = outdir_test / "test_out_cubes.txt"
    nPing = file_test.write_text("ping")
    assert nPing == 4

    l_tasks = list(range(25))

    num_threads = len (l_tasks)

    print(f"Launching on {num_threads} thread(s)")

    with Pool(num_threads) as p:
        l = p.map(sample_from_part, l_tasks)





