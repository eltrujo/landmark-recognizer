# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 12:27:55 2018

@author: Pablo
"""

from utils import choose_class
import numpy as np
import json
import matplotlib.pyplot as plt

"""
PLOT DATA
"""

video_name = 'Video4.mp4'
model_names = ['eigen', 'fisher']

path = 'C:/Users/Pablo/Google Drive/TFM/'

# Get real labels of frames
f2 = open(path + 'Videos/' + 'labels_real_' + video_name[:-4] + '.json', 'r')
real_data = json.load(f2)
f2.close()
real_data = np.array(map(lambda x: [x['class'], x['frameStart'], x['frameEnd']], real_data))

colors = ['b','m','r','k','y','c','g']

# Euclidean distance
fig = plt.figure(1)
plt.suptitle(video_name[:-4] + ' | Euclidean')

for i in range(len(model_names)):
    # Open json file and parse data to numpy array
    f = open(path + 'Videos/' + 'labels_' + model_names[i] + '_' + video_name[:-4] + '.json', 'r')
    data = json.load(f)
    f.close()
    num_frames = len(data)
    data = np.array(map(lambda x: [x['frame'], x['c1'], x['d1'], x['c2'], x['d2']], data))
    order = np.argsort(data, axis=0)[:,0] # order from initial frame to last frame
    data = data[order]
    max_dist1 = data[:,2].max()
    
    # Plot predicted labels and distances to nearest neighbors
    ax1 = plt.subplot('21'+str(i+1))
    ax1.plot(range(num_frames), data[:,2], color=(.5,.5,.5,.5), linewidth=.8)
    map(lambda x: ax1.plot(x, data[x,2], colors[int(data[x,1])] + '.'), range(num_frames))
    ax1.set_ylim(0, max_dist1)
    ax1.set_xlim(0, num_frames)
    ax1.set_xticks([])
    ax1.grid(True)
    ax1.set_title(model_names[i])
    
    # Plot real labels
    map(lambda x: ax1.plot(np.arange(x[1],x[2]+1), np.zeros(x[2]-x[1]+1, dtype=int), colors[x[0]] + 's'), real_data)
    
plt.get_current_fig_manager().window.showMaximized()
    
# Mahalanobis distance
fig = plt.figure(2)
plt.suptitle(video_name[:-4] + ' | Mahalanobis')

for i in range(len(model_names)):
    # Open json file and parse data to numpy array
    f = open(path + 'Videos/' + 'labels_' + model_names[i] + '_' + video_name[:-4] + '.json', 'r')
    data = json.load(f)
    f.close()
    num_frames = len(data)
    data = np.array(map(lambda x: [x['frame'], x['c1'], x['d1'], x['c2'], x['d2']], data))
    order = np.argsort(data, axis=0)[:,0] # order from initial frame to last frame
    data = data[order]
    max_dist2 = data[:,4].max()
    
    # Plot predicted labels and distances to nearest neighbors
    ax1 = plt.subplot('21'+str(i+1))
    ax1.plot(range(num_frames), data[:,4], color=(.5,.5,.5,.5), linewidth=.8)
    map(lambda x: ax1.plot(x, data[x,4], colors[int(data[x,3])] + '.'), range(num_frames))
    ax1.set_ylim(0, max_dist2)
    ax1.set_xlim(0, num_frames)
    ax1.set_xticks([])
    ax1.grid(True)
    ax1.set_title(model_names[i])
    
    # Plot real labels
    map(lambda x: ax1.plot(np.arange(x[1],x[2]+1), np.zeros(x[2]-x[1]+1, dtype=int), colors[x[0]] + 's'), real_data)
    
plt.get_current_fig_manager().window.showMaximized()


