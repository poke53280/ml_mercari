

#Texture attributes for detecting salt bodies in seismic data
#Tamir Hegazy
#∗ and Ghassan AlRegib
#Center for Energy and Geo Processing (CeGP), Georgia Institute of Technology

DATA_DIR_PORTABLE = "C:\\tgs_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(DATA_DIR + 'smooth_elf_post_img.jpg',0)


laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()


TRAIN_IMAGE_DIR = DATA_DIR + 'train/images/'
TRAIN_MASK_DIR = DATA_DIR + 'train/masks/'
TEST_IMAGE_DIR = DATA_DIR + 'test/images/'










train_fns = os.listdir(TRAIN_IMAGE_DIR)

X = [np.array(cv2.imread(TRAIN_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(train_fns)]