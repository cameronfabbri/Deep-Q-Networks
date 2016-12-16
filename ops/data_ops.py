import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
   Preprocess the image
   Resizes, crops, and grayscales the image
'''
def preprocess(img):
   cv2.imwrite('original.png', img)
   
   if len(img.shape) == 3:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   cv2.imwrite('gray_original.png', img)
   
   # resize
   img = cv2.resize(img, (84,110))
   cv2.imwrite('gray_resize.png', img)

   # crop
   img = img[110-84:110,:84]

   cv2.imwrite('gray_crop.png', img)
   return img
