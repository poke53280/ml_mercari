


import numpy as np
import pathlib
import cv2

from cv2 import CascadeClassifier




class HaarCascade():

    def __init__(self, path):
        
        assert path.is_dir()
        data_file = path / "haarcascade_frontalface_default.xml"
        assert data_file.is_file()

        self.face_cascade = cv2.CascadeClassifier(str(data_file))


    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces





