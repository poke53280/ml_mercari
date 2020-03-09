
#
# Investigate 
#
# p20: auyndsjryc.mpg
#
#
# Jumpiness
#

from VideoManager import VideoManager
from mp4_frames import get_part_dir
from mp4_frames import get_ready_data_dir
from face_detector import MTCNNDetector
from mp4_frames import read_video


from image_sampler import cut_frame
from featureline import find_all_face_boxes


from PIL import Image
import numpy as np

import matplotlib.pyplot as plt


v = VideoManager()

df = v._df

target = "plrtfgcwau.mp4"


# target = "wnhlvjdtfg.mp4"

m = df.file == target

if m.sum() == 0:
    print("Not registered")

assert m.sum() == 1

video = df[m].iloc[0]

original = video.orig
iCluster = video.cluster
part = video.part
file = video.file

l_video_set = list(df[df.orig == original].file)




print(f"Video: {file} Cluster: {iCluster} Original: {original} Part: {part}")


input_dir = get_part_dir(part)

assert (input_dir / file).is_file()
assert (input_dir / original).is_file()

video_real = read_video(input_dir / original, 128)
video_fake = read_video(input_dir / file, 128)

mtcnn_detector = MTCNNDetector()

l_faces_fake = find_all_face_boxes(mtcnn_detector, video_fake[:128])


x_max = video_fake.shape[2]
y_max = video_fake.shape[1]


for iFrame in range(video_fake.shape[0]):
    face = l_faces_fake[iFrame][0]

    bb_min = np.array(face['bb_min'])
    bb_max = np.array(face['bb_max'])

    eye_size = 0.2 * (bb_max - bb_min)

    l_eye_c = np.array(face['l_eye'])

    bb_min = l_eye_c - eye_size
    bb_max = l_eye_c + eye_size

    arScale = np.array([x_max, y_max])

    bb_min = (bb_min * arScale).astype(np.int32)
    bb_max = (bb_max * arScale).astype(np.int32)

    outputsize = 128 + 64

    im_mask, im_test = cut_frame(bb_min, bb_max, video_real, video_fake, iFrame, outputsize, False)

    im_test.save(get_ready_data_dir() / f"test_{iFrame:003}.png")
    im_mask.save(get_ready_data_dir() / f"test_{iFrame:003}_m.png")


    # def sharpness


    im = im_mask.convert('L') # to grayscale
    array = np.asarray(im, dtype=np.int32)

    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)

    print(f"Sharpness {sharpness}")


# run ImageAligner

l_data = []

for iFrame in range(127):
    im_test0 = Image.open(get_ready_data_dir() / f"cut_test_{iFrame:003}.png")
    im_test1 = Image.open(get_ready_data_dir() / f"cut_test_{iFrame + 1:003}.png")

    array0 = np.asarray(im_test0)
    array1 = np.asarray(im_test1)

    image_3 = np.sum((array1-array0)**2,axis=2)

    l_data.append(np.mean(image_3))


np.array(l_data).mean()
np.array(l_data).std()






im_A = Image.open(get_ready_data_dir() / f"test_004.png")
im_B = Image.open(get_ready_data_dir() / f"test_005.png")


f, axarr = plt.subplots(1,2)
axarr[0].imshow(im_A)
axarr[1].imshow(im_B)


plt.show()

