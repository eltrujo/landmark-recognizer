# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:36:15 2018

@author: Pablo
"""

import cv2
from imutils import rotate_bound
import json

path_vids = 'C:/Users/Pablo/Google Drive/TFM/Videos/'
path_ims = 'C:/Users/Pablo/Google Drive/TFM/Images from Videos/'
video_names = ['Video1.mp4', 'Video2.mp4', 'Video3.mp4', 'Video4.mp4']

for video_name in video_names:
    f = open(path_vids + 'labels_real_' + video_name[:-4] + '.json', 'r')
    data = json.load(f)
    f.close()
    data = sorted(data, key=lambda x: x['frameStart'])
    c = 0
    f_start = data[c]['frameStart']
    f_end = data[c]['frameEnd']
    
    # Load video
    cap = cv2.VideoCapture(path_vids + video_name)
    if not cap.isOpened():
        print('Error opening video file')
    
    w, h = (int(cap.get(3)), int(cap.get(4)))
    #w, h = (720, 1280)
    if w > h:
        transpose = True
        w, h = h, w
    else:
        transpose = False
    
    frame = 0
    count = 0
        
    while(True):
        # Read next frame
        st, im = cap.read()
        if im is None:
            break
        if frame >= f_start:
            if transpose:
                im = rotate_bound(im, 90)
            cv2.imwrite(path_ims + video_name[-5] + '-' + str(data[c]['class']) + '-' + str(count) + '.png', im)
            count += 1
        if frame > f_end:
            c += 1
            count = 0
            if c == len(data):
                break
            else:
                f_start = data[c]['frameStart']
                f_end = data[c]['frameEnd']
        frame+=1
    
    cap.release()
        
        