# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:39:38 2018

@author: Pablo
"""

import matplotlib.pyplot as plt
import numpy as np
import os

path = 'C:/Users/Pablo/Google Drive/TFM/Images/'

def change_color(vector):
        intensity = 381.0
        if vector.sum() == 0:
            return np.array([round(intensity/3)]*3, dtype=np.uint8)
        else:
            return (vector*intensity/vector.sum()).round().astype(np.uint8)
        
if __name__ == 'main':
    for pathnew in os.listdir(path):
        print(pathnew)
        for img_name in os.listdir(path + pathnew):
            im = plt.imread(path + pathnew + '/' + img_name)
            h, w, _ = im.shape
            im_flat = im.reshape(h*w, 3)
            
            new_im = np.array(map(change_color, im_flat)).reshape(h, w, 3)
            
            plt.imsave(path + '../Images 2D/' + pathnew + '/' + img_name, new_im)
