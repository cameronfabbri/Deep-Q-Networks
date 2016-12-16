import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
   Preprocess the image
   Resizes, crops, and grayscales the image
'''
def preprocess(img):
   if len(img.shape) == 3:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

   # resize
   img = cv2.resize(img, (84,110))

   # crop
   img = img[110-84:110,:84]

   return img
