


import cv2


DATA_DIR = "C:\\sm_match"

img = cv2.imread(DATA_DIR + "\\mat.jpg")

type (img)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


