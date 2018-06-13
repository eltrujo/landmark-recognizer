# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:11:04 2018

@author: Pablo
"""

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from imutils import rotate_bound
import pdb

video_name = 'Video5.mp4'

path = 'C:/Users/Pablo/Google Drive/TFM/Videos/'

cap = cv2.VideoCapture(path + video_name)
if not cap.isOpened():
    print('Error opening video file')

w, h = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
if w > h:
    out = cv2.VideoWriter(path + video_name[:-4] + '_' + 'FrameNumber'  + '.avi', fourcc, round(cap.get(5),1), (h,w), True)
    transpose = True
else:
    out = cv2.VideoWriter(path + video_name[:-4] + '_' + 'FrameNumber'  + '.avi', fourcc, round(cap.get(5),1), (w,h), True)
    transpose = False

fnt = ImageFont.truetype("fonts/stocky.ttf",100)

frame = 0

while(True):
    _, im = cap.read()
    if im is None:
        break
    if transpose:
        im = rotate_bound(im, 90)
    im = Image.fromarray(im, 'RGB')
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    draw = ImageDraw.Draw(im)
    draw.text((20, 10), str(frame), (255,0,0), font=fnt)
    
    r, g, b = im.split()
    im = Image.merge("RGB", (b, g, r))
    im = np.array(im, dtype=np.uint8)
    
    out.write(im)
    
    frame += 1
    
cap.release()
out.release()