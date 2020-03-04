

import numpy as np
from mp4_frames import get_part_dir
from mp4_frames import read_video
from mp4_frames import read_metadata

from featureline import find_middle_face_box
from face_detector import MTCNNDetector
from line_sampler import get_line

from sklearn.metrics import mean_squared_error


a = [7, 3, 4, 11, 24, 123, 3,  7, 3,  4, 10,  21, 123, 3]
b = [7, 3, 4, 11, 24, 129, 6, 11, 4, 10, 21, 123,   3, 3]


# eye - eye line.
# Start .5 outside, continue to l_eye location.

l_d = read_metadata(7)

entry = l_d[3]

input_dir = get_part_dir(7)
mtcnn_detector = MTCNNDetector()

    
real_path = input_dir / entry[0]
fake_path = input_dir / entry[1][1]

assert real_path.is_file()
assert fake_path.is_file()

video_size = 32

W = 256
H = 1

real_video = read_video(real_path, video_size)
fake_video = read_video(fake_path, video_size)

x_max = real_video.shape[2]
y_max = real_video.shape[1]
z_max = real_video.shape[0]



face = find_middle_face_box(mtcnn_detector, real_video)


x0 = face[0][1]
x1 = face[1][1]

y0 = face[0][0]
y1 = face[1][0]

real_image = real_video[z_max // 2]

fake_image = fake_video[z_max // 2]

real_face = real_image[x0 : x1, y0: y1, :]
fake_face = fake_image[x0 : x1, y0: y1, :]

plt.imshow(real_face)

plt.show()

plt.imshow(fake_face)
plt.show()

lx0 = x0 - real_face.shape[0]
lx1 = x0 + real_face.shape[0] // 2

ly0 = y0
ly1 = y0

l = get_line(np.array([lx0,ly0,0]), np.array([lx1, ly1, z_max -1]))

l = np.swapaxes(l, 0, 1)

l_z = l[:, 2].astype(np.int32)
l_y = l[:, 1].astype(np.int32)
l_x = l[:, 0].astype(np.int32)

diff_video = real_video[l_z, l_y, l_x].astype(np.float32) - fake_video[l_z, l_y, l_x]

image_3 = np.sqrt(np.sum(1.0 * (diff_video**2),axis=1))

plt.plot(list(image_3))
plt.show()

diff * diff




df = pd.DataFrame({'id' : [0,1], 'v' : [31, 39]})

df_p = pd.DataFrame({'id' : [0, 0, 0, 0, 0, 1, 1, 1, 1], 'ft' : [3, 4, 5, 1, 3, 5, 3, 9, 1]})

df_m = df.merge

g = df_p.groupby('id')['ft']

g.nth(4)

