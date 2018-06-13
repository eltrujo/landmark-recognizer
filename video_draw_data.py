# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 17:06:15 2018

@author: Pablo
"""

import cv2
import os
from utils import choose_class
import numpy as np
import json
import matplotlib.pyplot as plt
import utils
import imutils
import pdb
import time

"""
DRAW DATA INTO VIDEO AND DISPLAY IT
"""

video_name = 'Video4.mp4'
model_name = 'eigen'

plt.ion() # Interactive figure activated
path = 'C:/Users/Pablo/Google Drive/TFM/'

if model_name == 'eigen':
    num_components = 40
elif model_name == 'fisher':
    num_components = 6
else:
    print('ERROR: Wrong model_name')

# Open json file and parse data to numpy array
f = open(path + 'Videos/' + 'labels_' + model_name + '_' + video_name[:-4] + '.json', 'r')
data = json.load(f)
f.close()
num_frames = len(data)
data = np.array(map(lambda x: [x['frame'], x['c1'], x['d1'], x['c2'], x['d2']], data))
order = np.argsort(data, axis=0)[:,0] # order from initial frame to last frame
data = data[order]
max_dist1 = data[:,2].max()
max_dist2 = data[:,4].max()

# Load video
cap = cv2.VideoCapture(path + 'Videos/' + video_name)
if not cap.isOpened():
    print('Error opening video file')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(video_name[:-4] + '_' + model_name + '_DRAW.avi', fourcc, 30.0, (1000,700), True)
    
    
# Load model
if model_name == 'eigen':
    model = cv2.face.createEigenFaceRecognizer()
elif model_name == 'fisher':
    model = cv2.face.createFisherFaceRecognizer()
model.load(path + 'Models/' + model_name + '.yml')

eigen_imgs = model.getEigenVectors()
mean_vector = model.getMean()

# Draw data on video
frame = 0
w, h = (int(cap.get(3)), int(cap.get(4)))
#w, h = (720, 1280)
if w > h:
    transpose = True
else:
    transpose = False

colors = ['b','m','r','k','y','c','g']

fig = plt.figure(1)
fig.set_size_inches(10,7)

def build_figure():
    plt.figure(1)
    
    # Prepare subplots with different sizes
    ax = plt.subplot2grid((2,3), (0,0), rowspan=2)
    ax.axis('off')
    
    ax1 = plt.subplot2grid((2,3), (0,1))
    ax1.plot(range(num_frames), data[:,2], color=(.5,.5,.5,.5), linewidth=.8)
    map(lambda x: ax1.plot(x, data[x,2], colors[int(data[x,1])] + '.'), range(num_frames))
    ax1.set_ylim(0, max_dist1)
    ax1.set_xticks([])
    ax1.grid(True)
    
    ax2 = plt.subplot2grid((2,3), (1,1))
    ax2.plot(range(num_frames), data[:,4], color=(.5,.5,.5,.5), linewidth=.8)
    map(lambda x: ax2.plot(x, data[x,4], colors[int(data[x,3])] + '.'), range(num_frames))
    ax2.set_ylim(0, max_dist2)
    ax2.set_xticks([])
    ax2.grid(True)
    
    ax3 = plt.subplot2grid((2,3), (0,2), rowspan=2)
    ax3.axis('off')
    im = plt.imread(path + 'legend.png')
    ax3.imshow(im)
    return ax, ax1, ax2

# Prepare subplots
ax, ax1, ax2 = build_figure()

while(True):
    print(frame)
    
    # Read next frame
    st, im = cap.read()
    if im is None:
        break
    if transpose:
        im = imutils.rotate_bound(im, 90)
    
    # Get class and distance to closest neighbour
    class1, dist1, class2, dist2 = data[frame,1:]
    class1 = utils.choose_class(int(class1))
    class2 = utils.choose_class(int(class2))
    
    # Display new frame and forward plots
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax1.set_xlim(frame, frame+60)
    ax1.set_title('Eucl. | ' + class1)
    ax2.set_xlim(frame, frame+60)
    ax2.set_title('Mahal. | ' + class2)
    
    # Get figure as image and save it to video
    img = utils.fig2data(fig)
    out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    frame += 1
    if frame % 36 == 0:
        fig.clf()
        ax, ax1, ax2 = build_figure()
    
cap.release()
out.release()
    
