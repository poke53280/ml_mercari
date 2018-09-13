

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
from sklearn import linear_model

img = cv2.imread(DATA_DIR + 'smooth_elf_post_img.jpg',0)


laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

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

img = img.astype(np.int32)



def get_score(img, x0, y0, x1, y1):

    window = img[x0:x1, y0:y1]

    w_x = np.gradient(window, axis = 0)
    w_y = np.gradient(window, axis = 1)

    w_x = w_x.flatten()
    w_y = w_y.flatten()

    #plt.scatter(w_x, w_y)

    #plt.show()

    X = w_x
    y = w_y

    lm = linear_model.LinearRegression()

    model = lm.fit(X.reshape(-1,1),y)
    score = model.score(X.reshape(-1,1),y)

    print(f"Score = {score}")

"""c"""


# Non salt
1314, 574 , 1470, 707
330, 240, 490, 365
1247, 1066, 1370, 1160

# Salt
788, 922,   918, 1040
822, 657,  914, 763


get_score(img, 1314, 574 , 1470, 707)
# => 0.12
get_score(img, 330, 240, 490, 365)
# => 0.0002

get_score(img, 1247, 1066, 1370, 1160)
# 0.07

get_score(img, 788, 922,   918, 1040)



train_fns = os.listdir(TRAIN_IMAGE_DIR)

X = [np.array(cv2.imread(TRAIN_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(train_fns)]