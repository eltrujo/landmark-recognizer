# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:28:11 2018

@author: Pablo
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle
import utils
from imutils import rotate_bound
import json
from pdb import set_trace as STOP

technique = 'orb'
video_names = ['Video1.mp4', 'Video2.mp4', 'Video3.mp4', 'Video4.mp4']


def extract_features_ORB(images, orb):
    descriptors = []
    for img in images:
        descriptors.append(orb.detectAndCompute(img, None)[1])
    return descriptors

# =============================================================================
# LOAD IMAGES
# =============================================================================
path = 'C:/Users/Pablo/Google Drive/TFM/'

# Load images
class_names = os.listdir(path + 'Images/')
class_numbers = list(map(lambda x: int(x[0]), class_names))
images, labels = utils.load_images_and_labels(path + 'Images/')
num_samples, height, width = images.shape

MAX_MATCHES_array = np.arange(150,601,50)
accuracies = np.zeros((len(video_names),len(MAX_MATCHES_array)))

for index, MAX_MATCHES in enumerate(MAX_MATCHES_array):
    
    # =============================================================================
    # CREATE / LOAD ORB FEATURES
    # =============================================================================
    path = 'C:/Users/Pablo/Google Drive/TFM/'
    orb = cv2.ORB_create(MAX_MATCHES)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    if not os.path.isfile(path + 'Models/orb_' + str(MAX_MATCHES) + '.pkl'):
        # Detect ORB features and compute descriptors
        descriptors = extract_features_ORB(images, orb) # each image has MAX_MATCHES descriptors with 32 items each
        pickle.dump(descriptors, open(path + 'Models/orb_' + str(MAX_MATCHES) + '.pkl', 'wb'))
    else:
        descriptors = pickle.load(open(path + 'Models/orb_' + str(MAX_MATCHES) + '.pkl', 'rb'))
        
    # =============================================================================
    # PREDICT THE LABELS FOR EACH FRAME AND SAVE IT TO A JSON FILE
    # =============================================================================
    path = 'C:/Users/Pablo/Google Drive/TFM/Videos/'
    
    for video_name in video_names:
        # Open json file to store data
        f = open(path + 'labels_' + technique + '_' + str(MAX_MATCHES) + '_' + video_name[:-4] + '.json', 'w')
        keys = ["frame", "c1", "d1"]
        f.write('[\n')
        
        # Load video
        cap = cv2.VideoCapture(path + video_name)
        if not cap.isOpened():
            print('Error opening video file')
        
        w, h = (int(cap.get(3)), int(cap.get(4)))
        #w, h = (720, 1280)
        if w > h:
            transpose = True
        else:
            transpose = False
            
        frame = 0
        
        # Store first frame's prediction
        st, im = cap.read()
        if im is not None:
            if transpose:
                im = rotate_bound(im, 90)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            desc = extract_features_ORB([im], orb)[0]
            dists = map(lambda x: matcher.match(desc, x, None), descriptors)
            dists = list(map(lambda x: sum(y.distance for y in x)/len(x), dists))
            neigh = np.argmin(dists)
            info = [frame, labels[neigh], round(dists[neigh],2)]
            f.write(str(dict(zip(keys, info))).replace("'", '"'))
            frame += 1
            
        while(True):
            # Read next frame
            st, im = cap.read()
            if im is None:
                break
            if transpose:
                im = rotate_bound(im, 90)
            
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            desc = extract_features_ORB([im], orb)[0]
            if desc is not None:
                if len(desc) >= MAX_MATCHES*.75:
                    dists = map(lambda x: matcher.match(desc, x, None), descriptors)
                    dists = list(map(lambda x: sum(y.distance for y in x)/len(x), dists))
                    neigh = np.argmin(dists)
                    info = [frame, labels[neigh], round(dists[neigh],2)]
            else:
                info = [frame, labels[neigh], round(dists[neigh]*2,2)]
            f.write(',\n' + str(dict(zip(keys, info))).replace("'", '"'))
            frame += 1
        
        f.write('\n]')
        f.close()
        cap.release()
        
    # =============================================================================
    # COMPARE DATA WITH REAL VALUES
    # =============================================================================
    print('Featrures: ' + str(MAX_MATCHES))
    for i, video_name in enumerate(video_names):
        # Get real labels of frames
        f2 = open(path + 'labels_real_' + video_name[:-4] + '.json', 'r')
        real_data = json.load(f2)
        f2.close()
        real_data = np.array(list(map(lambda x: [x['class'], x['frameStart'], x['frameEnd']], real_data)))
        
        # Get predicted labels of frames
        f = open(path + 'labels_' + technique + '_' + str(MAX_MATCHES) + '_' + video_name[:-4] + '.json', 'r')
        data = json.load(f)
        f.close()
        num_frames = len(data)
        data = np.array(list(map(lambda x: [x['frame'], x['c1'], x['d1']], data)))
        order = np.argsort(data, axis=0)[:,0] # order from initial frame to last frame
        data = data[order]
        
        total_real_frames = 0
        predicted_frames = 0
        for elem in real_data:
            total_real_frames += elem[2] - elem[1] + 1
            for j in range(elem[1], elem[2]+1):
                if data[j,1] == elem[0]:
                    predicted_frames += 1
        print('Accuracy: ' + str(round(100*float(predicted_frames)/total_real_frames,2)) + ' %')
        accuracies[i,index]
    # =============================================================================
    # PLOT DATA AND COMPARE IT WITH REAL VALUES
    # =============================================================================
    
    colors = ['b','m','r','k','y','c','g']
    
    fig = plt.figure('ORB: ' + str(MAX_MATCHES))
    plt.suptitle('ORB: ' + str(MAX_MATCHES))
    
    for i, video_name in enumerate(video_names):
        # Get real labels of frames
        f2 = open(path + 'labels_real_' + video_name[:-4] + '.json', 'r')
        real_data = json.load(f2)
        f2.close()
        real_data = np.array(list(map(lambda x: [x['class'], x['frameStart'], x['frameEnd']], real_data)))
        
        # Get predicted data
        f = open(path + 'labels_' + technique + '_' + str(MAX_MATCHES) + '_' + video_name[:-4] + '.json', 'r')
        data = json.load(f)
        f.close()
        num_frames = len(data)
        data = np.array(list(map(lambda x: [x['frame'], x['c1'], x['d1']], data)))
        order = np.argsort(data, axis=0)[:,0] # order from initial frame to last frame
        data = data[order]
        max_dist1 = data[:,2].max()
        
        # Plot predicted labels and distances to nearest neighbors
        ax1 = plt.subplot('22'+str(i+1))
        ax1.plot(range(num_frames), data[:,2], color=(.5,.5,.5,.5), linewidth=.8)
        for frame in range(num_frames):
            ax1.plot(frame, data[frame,2], colors[int(data[frame,1])] + '.')
        ax1.set_ylim(0, max_dist1)
        ax1.set_xlim(0, num_frames)
        ax1.set_xticks([])
        ax1.grid(True)
        
        # Plot real labels
        for x in real_data:
            ax1.plot(np.arange(x[1], x[2]+1), np.zeros(x[2]-x[1]+1, dtype=int), colors[x[0]] + 's')
        
    plt.get_current_fig_manager().window.showMaximized()
    
# =============================================================================
# PLOT ACCURACY FOR EACH VIDEO
# =============================================================================
fig = plt.figure()
plt.suptitle('Accuracies')
for i in range(len(video_names)):
    ax = plt.subplot('22'+str(i+1))
    ax.plot(MAX_MATCHES_array, accuracies[i], 'b', marker='o')
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.grid()
plt.show()
