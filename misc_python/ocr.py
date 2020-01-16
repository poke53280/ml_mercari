


import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd

from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import cv2 

pytesseract.get_tesseract_version()

import seaborn as sns
sns.set(color_codes=True)


image_png = Image.open("C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\sm_date.png")

img_color = np.array(image_png)

num_pixs = img_color.shape[0] * img_color.shape[1]

m0 = (img_color[:, :, 0] == 255) & (img_color[:, :, 1] == 255) & (img_color[:, :, 2] == 255) & (img_color[:, :, 3] == 255)
m1 = (img_color[:, :, 0] == 0)   & (img_color[:, :, 1] == 0)   & (img_color[:, :, 2] == 0)   & (img_color[:, :, 3] == 255)

assert m0.sum() + m1.sum() == m0.shape[0] * m0.shape[1]

img_gray = np.empty((img_color.shape[0], img_color.shape[1], 1), dtype = np.uint8)

img_gray[m0] = 255
img_gray[m1] = 0



# Apply edge detection method on the image 
edges = cv2.Canny(img_gray,50,150,apertureSize = 3)

cv2.imshow('Color image',img_color)
cv2.imshow('Gray image',img_gray)
cv2.imshow('edges image', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()


# This returns an array of r and theta values 

wiggle = 0.1

direction = np.pi/2.0

min_theta = direction - wiggle
max_theta = direction + wiggle

lines_0 = cv2.HoughLines(image = edges,
                       rho = 1,
                       theta = np.pi/(10 * 180),
                       threshold = 520,
                       lines = 0,
                       srn = 0,
                       stn = 0,
                       min_theta = min_theta,
                       max_theta = max_theta)


lines_0 = np.squeeze(lines_0)

print (lines_0.shape)


img_lines = img_color.copy()


for r,theta in lines_0: 
      
    a = np.cos(theta) 
    b = np.sin(theta) 
      
    x0 = a*r 
    y0 = b*r 
      
    x1 = int(x0 + 3000*(-b)) 
    y1 = int(y0 + 3000*(a)) 
  
    x2 = int(x0 - 3000*(-b)) 
    y2 = int(y0 - 3000*(a)) 
      
    cv2.line(img_lines,(x1,y1), (x2,y2), (0,255, 0) ,2)
      

cv2.imshow('image0',img_color)
cv2.imshow('lines',img_lines)


cv2.waitKey(0)
cv2.destroyAllWindows()


acAngles = lines[:, 1]

# Histogram
sns.distplot(acAngles)
plt.show()

rot_degrees = np.rad2deg(np.mean(acAngles)) - 90


image_center = (0, 0)

rot_mat = cv2.getRotationMatrix2D(image_center, rot_degrees, 1.0)


rotated_color_image = cv2.warpAffine(img_color, rot_mat, img_color.shape[1::-1], flags=cv2.INTER_LINEAR)


cv2.imshow('image_base', img_color)
cv2.imshow('image_rotated', rotated_color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# OCR IMAGE




image_png = Image.open("C:\\Users\\T149900\\source\\repos\\PythonApplication2\\PythonApplication2\\sm_date.png")

img_color = np.array(image_png)


newimg = cv2.resize(img_color,(int(200),int(200)))

# cv2.imshow('image_base', img_color)
cv2.imshow('image_large', newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


erode_ = (3, 3)


eroded = cv2.erode(src = newimg, kernel = erode_, iterations = 10)



dilate_=(3, 3)


id = cv2.dilate(src = eroded, kernel = dilate_, iterations = 10)

cv2.imshow('newimg', newimg)
cv2.imshow('eroded', eroded)
cv2.imshow('dilated', id)
cv2.waitKey(0)
cv2.destroyAllWindows()


i = Image.fromarray(eroded)

txt = pytesseract.image_to_string(i, lang = 'nor', nice = 1)

print (txt)





