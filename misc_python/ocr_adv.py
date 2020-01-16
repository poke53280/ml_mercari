

# https://stackoverflow.com/questions/33881175/remove-background-noise-from-image-to-make-text-more-clear-for-ocr
# http://www.robindavid.fr/opencv-tutorial/cracking-basic-captchas-with-opencv.html


import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd

from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import cv2 as cv

pytesseract.get_tesseract_version()

import seaborn as sns
sns.set(color_codes=True)


im = cv.imread('C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\aRh8C.png', cv.CV_8UC1)



# apply Otsu threshold

ret, bw = cv.threshold(im, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

dist = cv.distanceTransform(bw, cv.DIST_L2, cv.DIST_MASK_PRECISE)


# threshold the distance transformed image
SWTHRESH = 8    # stroke width threshold

ret, dibw = cv.threshold(dist, SWTHRESH/2, 255, cv.THRESH_BINARY)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

# perform opening, in case digits are still connected

morph = cv.morphologyEx(dibw, cv.MORPH_OPEN, kernel)

dibw = dibw.astype(np.uint8)


# find contours and filter
cont = morph.astype(np.uint8)


binary = cv.cvtColor(dibw, cv.COLOR_GRAY2BGR);

HTHRESH = int(im.shape[0] * .5)

# https://docs.opencv.org/master/d3/d05/tutorial_py_table_of_contents_contours.html
contours, hierarchy = cv.findContours(cont, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, None, None, (0,0))

im_c = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
cont_image = cv.drawContours(im_c, contours, -1, (255,0,0), thickness=2)


for c in contours:
    rect = cv.minAreaRect(c)

    box = cv.boxPoints(rect)
    box = np.int0(box)
    cont_image = cv.drawContours(cont_image,[box],0,(0,0,255),2)




#cv.boundingRect(contours[0])
#hierarchy[:, 0]



cv.imshow('bw', bw)
cv.imshow('dist', dist)
cv.imshow('dibw', dibw)
cv.imshow('morph', morph)
cv.imshow('cont_image', cont_image)
cv.waitKey(0)
cv.destroyAllWindows()










opencvImage = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2GRAY)


# Apply edge detection method on the image 
edges = cv2.Canny(gray,50,150,apertureSize = 3)

cv2.imshow('Original image',opencvImage)
cv2.imshow('Gray image', gray)
cv2.imshow('edges image', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()


# This returns an array of r and theta values 

wiggle = 0.1

direction = np.pi/2.0

min_theta = direction - wiggle
max_theta = direction + wiggle


lines = cv2.HoughLines(edges,1,np.pi/(10 * 180), 320, 0, 0, 0, min_theta, max_theta)
lines = np.squeeze(lines)


#m = lines[:, 1 ] < 0.1
#lines = lines[m]

lines


print (lines.shape)

opencvImage = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR)
for r,theta in lines: 
      
    a = np.cos(theta) 
    b = np.sin(theta) 
      
    x0 = a*r 
    y0 = b*r 
      
    x1 = int(x0 + 3000*(-b)) 
    y1 = int(y0 + 3000*(a)) 
  
    x2 = int(x0 - 3000*(-b)) 
    y2 = int(y0 - 3000*(a)) 
      
    cv2.line(opencvImage,(x1,y1), (x2,y2), (0,255, 0) ,2)
      

cv2.imshow('image',opencvImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


acAngles = lines[:, 1]

# Histogram
sns.distplot(acAngles)
plt.show()

rot_degrees = np.rad2deg(np.mean(acAngles)) - 90


image_center = (0, 0)

rot_mat = cv2.getRotationMatrix2D(image_center, rot_degrees, 1.0)

opencvImage = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR)
result = cv2.warpAffine(opencvImage, rot_mat, opencvImage.shape[1::-1], flags=cv2.INTER_LINEAR)


cv2.imshow('image',result)
cv2.waitKey(0)
cv2.destroyAllWindows()



# OCR IMAGE

txt = pytesseract.image_to_string(i, lang = 'nor', nice = 1)

print (txt)




