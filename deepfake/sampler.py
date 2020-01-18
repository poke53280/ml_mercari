

import numpy as np


def get_line(p0, p1):

    dp = p1 - p0
    dp = np.abs(dp)

    num_steps = np.max(dp)

    # t element of [0, 1]

    step_size = 1 / num_steps

    ai = np.arange(start = 0, stop = 1 + step_size, step = step_size)

    ai_t = np.tile(ai, 3).reshape(-1, ai.shape[0])


    p = (p1 - p0).reshape(3, -1) * ai_t

    p = p + p0.reshape(3, -1)

    p = np.round(p)

    return p



# input 'video'

def get_lines_from_video(video, N):


    x0 = 0
    x1 = video.shape[0]

    y0 = 0
    y1 = video.shape[1]

    z0 = 0
    z1 = video.shape[2]

    data = np.zeros((N, 16, 3))


    # Sample from volumne

    iRerun = 0

    for i in range(N):

        f_min = np.random.choice(x1 - x0)
        f_max = np.random.choice(x1 - x0)

        x0_min = 33  # b[f_min].x0
        x1_min = 39  ....




        p0 = np.array([x0 + np.random.choice(x1 - x0), y0 + np.random.choice(y1 - y0), z0 + np.random.choice(z1 - z0)])
        p1 = np.array([x0 + np.random.choice(x1 - x0), y0 + np.random.choice(y1 - y0), z0 + np.random.choice(z1 - z0)])

        while np.square(p1 - p0).sum() < (16 * 16 * 16):
            iRerun = iRerun + 1
            p1 = np.array([x0 + np.random.choice(x1 - x0), y0 + np.random.choice(y1 - y0), z0 + np.random.choice(z1 - z0)])

        l = get_line(p0, p1)

        assert l.shape[1] >= 16

        l = np.swapaxes(l, 0, 1)

        l = l[:16]
        l = l.astype(np.int32)

        l_x = l[:, 0]
        l_y = l[:, 1]
        l_z = l[:, 2]

        data[i] = video[l_x, l_y, l_z]

    
    rOverScan = iRerun/ N
    print(f"over scan {rOverScan}")

    return data


