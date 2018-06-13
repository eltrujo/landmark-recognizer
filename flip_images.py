# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:04:12 2018

@author: Pablo
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob2 import glob
import os
from pdb import set_trace as STOP

path = 'C:/Users/Pablo/Desktop/AAA/'
folders = ['Modified Images/', 'Modified Images 2D/',
           'Modified Images Reduced/', 'Modified Images SKF/']

def modify_images(path):
    """
    path: folder (Images) that contains subfolders with the name of the class of the images inside each of them
    path type: string
        (Change image size), (convert to grayscale) and save them in folder Modified Images or Reduced Images
    """
    save_path = path + '../Modified Images 1/'
#    save_path = path + '../Modified Images Gray/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_dir_names = os.listdir(path)
    for current_dir_name in all_dir_names:
        if not os.path.exists(save_path + current_dir_name):
            os.makedirs(save_path + current_dir_name)
        img_names = os.listdir(path + current_dir_name)
        for i in range(len(img_names)):
            # Reduce size
#            im = cv2.resize(cv2.imread(path + current_dir_name + '/' + img_names[i]), (90,160))
            # Reduce size and convert to grayscale
            im = cv2.cvtColor(cv2.resize(cv2.imread(path + current_dir_name + '/' + img_names[i]), (90,160)), cv2.COLOR_BGR2GRAY)
            # Convert to grayscale
#            im = cv2.cvtColor(cv2.imread(path + current_dir_name + '/' + img_names[i]), cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path + current_dir_name + '/' + str(i+1) + '.png', im)
    
    print('Images modified')

path = 'C:/Users/Pablo/Desktop/AAA/Images/'
modify_images(path)

#for folder in folders:
#    print(folder)
#    for label in glob(path + folders[0] + '/*'):
#        print(label)
#        for filename in glob(label + '/*.png'):
#            im = plt.imread(filename)
#            STOP()
#            plt.imsave(filename, np.fliplr(im))
            

