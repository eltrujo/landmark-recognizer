# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:33:10 2018

@author: Pablo
link: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
Blurrines is measured as the variation of the Laplacian kernel's convolution
"""

import os
import numpy as np
from glob2 import glob
import cv2

path = 'C:/Users/Pablo/Google Drive/TFM/Images/0 - Classroom/'

im = cv2.imread(path + '1.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(im, cv2.CV_64F)
cv2.imwrite('C:/Users/Pablo/Desktop/1.png', lap)
print(cv2.Laplacian(im, cv2.CV_64F).var())

im2 = cv2.imread(path + '2.png')
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
lap2 = cv2.Laplacian(im2, cv2.CV_64F)
cv2.imwrite('C:/Users/Pablo/Desktop/2.png', lap2)
print(cv2.Laplacian(im2, cv2.CV_64F).var())



#img_names = glob(path + '/*.png')
#how_blur = np.zeros(len(img_names))
#for i, img_name in enumerate(img_names):
#    im = cv2.imread(img_name)
#    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#    how_blur[i] = cv2.Laplacian(im, cv2.CV_64F).var() # the smaller the blurrier
#indexes = np.argsort(how_blur)


#for this_dir in os.listdir(path):
#    img_names = glob(path + this_dir + '/*.png')
#    images = []
#    for im_name in img_names:
#        images.append(cv2.imread(im_name))
#        os.remove(im_name)
#    for i in range(len(images)):
#        cv2.imwrite(path + this_dir + '/' + str(i+1) + '.png', images[i])