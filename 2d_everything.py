# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:11:39 2018

@author: Pablo
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import utils
from sklearn.neighbors import KNeighborsClassifier
import pickle
from imutils import rotate_bound
import json
from pdb import set_trace as STOP

technique = '2d'
video_names = ['Video1.mp4', 'Video2.mp4', 'Video3.mp4', 'Video4.mp4']
#model_name = 'eigen'
model_name = 'fisher'


def change_color(flat_im):
    sum_array = np.sum(flat_im, axis=1, keepdims=True)
    sum_array[sum_array == 0] = 1
    im = flat_im * 381.0 / sum_array
    return np.round(im[:,:-1]).astype(np.uint8)
    
# =============================================================================
# LOAD IMAGES AS FLATTENED ARRAY OF TWO IMAGE CHANNELS
# =============================================================================
path = 'C:/Users/Pablo/Google Drive/TFM/'

# Load images
class_names = os.listdir(path + 'Modified Images 2D/')
class_numbers = list(map(lambda x: int(x[0]), class_names))
images, labels = utils.load_images_and_labels_2d(path + 'Modified Images 2D/')
num_samples, height, width = images.shape

if model_name == 'eigen':
    num_components_array = np.arange(50, 201, 10)
elif model_name == 'fisher':
    num_components_array = np.arange(6,7,1)
accuracies_eucl = np.zeros((len(video_names),len(num_components_array)), dtype=np.float)
accuracies_mahal = np.zeros((len(video_names),len(num_components_array)), dtype=np.float)

for index, num_components in enumerate(num_components_array):
    
    path = 'C:/Users/Pablo/Google Drive/TFM/'
    
    # =============================================================================
    # CREATE / LOAD EIGEN RECOGNIZER
    # =============================================================================
        
    # Generate and save model or Load model if exists
    if not os.path.isfile(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml'):
        if model_name == 'eigen':
            model = cv2.face.EigenFaceRecognizer_create(num_components)
        elif model_name == 'fisher':
            model = cv2.face.FisherFaceRecognizer_create()
        model.train(images, labels)
        model.write(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml')
    else:
        if model_name == 'eigen':
            model = cv2.face.EigenFaceRecognizer_create()
        elif model_name == 'fisher':
            model = cv2.face.FisherFaceRecognizer_create()
        model.read(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml')
    
    # Get eigenvectors, mean and projections
    eigen_imgs = model.getEigenVectors()
    mean_vector = model.getMean()
    projections = np.array(model.getProjections()).reshape(num_samples, num_components)
    classes = model.getLabels().ravel()
    
    # =============================================================================
    # CREATE / LOAD K-NN CLASSIFIERS (EUCLIDEAN AND MAHALANOBIS METRIC)
    # =============================================================================
    
    ## Generate and save classifiers or Load classifiers if exist
    if not os.path.isfile(path + 'Models/' + model_name + '_K-NN_euclidean_' + technique + '_' + str(num_components) + '.sav'):
        classif1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        classif1.fit(projections, classes)
        pickle.dump(classif1, open(path + 'Models/' + model_name + '_K-NN_euclidean_' + technique + '_' + str(num_components) + '.sav', 'wb'))
    else:
        classif1 = pickle.load(open(path + 'Models/' + model_name + '_K-NN_euclidean_' + technique + '_' + str(num_components) + '.sav', 'rb'))
    if not os.path.isfile(path + 'Models/' + model_name + '_K-NN_mahalanobis_' + technique + '_' + str(num_components) + '.sav'):
        classif2 = KNeighborsClassifier(n_neighbors=1, metric='mahalanobis', metric_params=dict(V=np.cov(projections, rowvar=False)))
        classif2.fit(projections, classes)
        pickle.dump(classif2, open(path + 'Models/' + model_name + '_K-NN_mahalanobis_' + technique + '_' + str(num_components) + '.sav', 'wb'))
    else:
        classif2 = pickle.load(open(path + 'Models/' + model_name + '_K-NN_mahalanobis_' + technique + '_' + str(num_components) + '.sav', 'rb'))
        
    # =============================================================================
    # PREDICT THE LABELS FOR EACH FRAME AND SAVE IT TO A JSON FILE
    # =============================================================================
    path = 'C:/Users/Pablo/Google Drive/TFM/Videos/'
    
    for video_name in video_names:
        # Open json file to store data
        f = open(path + 'labels_' + model_name + '_' + video_name[:-4] + '_' + technique + '_' + str(num_components) + '.json', 'w')
        keys = ["frame", "c1", "d1", "c2", "d2"]
        f.write('[\n')
        
        # Load video
        cap = cv2.VideoCapture(path + video_name)
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
        
        # Store first frame's prediction
        st, im = cap.read()
        if im is not None:
            if transpose:
                im = rotate_bound(im, 90)
            im = cv2.resize(im, (90,160)).reshape(14400, 3)
            im = change_color(im).reshape(1,28800)
            im_projected = np.dot(eigen_imgs.T, (im-mean_vector).T).reshape(1,num_components)
            dist1, ind1 = classif1.kneighbors(X=im_projected, n_neighbors=1, return_distance=True)
            class1 = classes[ind1[0][0]]
            dist2, ind2 = classif2.kneighbors(X=im_projected, n_neighbors=1, return_distance=True)
            class2 = classes[ind2[0][0]]
            info = [frame, class1, round(dist1[0][0]), class2, round(dist2[0][0],3)]
            f.write(str(dict(zip(keys, info))).replace("'", '"'))
            frame += 1
            
        while(True):
            # Read next frame
            st, im = cap.read()
            if im is None:
                break
            if transpose:
                im = rotate_bound(im, 90)
            
            # Reduce size, convert to isoluminant and project to subspace
            im = cv2.resize(im, (90,160)).reshape(14400, 3)
            im = change_color(im).reshape(1,28800)
            im_projected = np.dot(eigen_imgs.T, (im-mean_vector).T).reshape(1,num_components)
            
            # Get nearest neighbor (euclidean and mahalanobis metrics)
            dist1, ind1 = classif1.kneighbors(X=im_projected, n_neighbors=1, return_distance=True)
            class1 = classes[ind1[0][0]]
            dist2, ind2 = classif2.kneighbors(X=im_projected, n_neighbors=1, return_distance=True)
            class2 = classes[ind2[0][0]]
            
            # Store data into json file
            info = [frame, class1, round(dist1[0][0]), class2, round(dist2[0][0],3)]
            f.write(',\n' + str(dict(zip(keys, info))).replace("'", '"'))
            frame += 1
        
        f.write('\n]')
        f.close()
        cap.release()
        
    # =============================================================================
    # COMPARE DATA WITH REAL VALUES
    # =============================================================================
    for i, video_name in enumerate(video_names):
        # Get real labels of frames
        f2 = open(path + 'labels_real_' + video_name[:-4] + '.json', 'r')
        real_data = json.load(f2)
        f2.close()
        real_data = np.array(list(map(lambda x: [x['class'], x['frameStart'], x['frameEnd']], real_data)))
        
        # Get predicted labels of frames
        f = open(path + 'labels_' + model_name + '_' + video_name[:-4] + '_' + technique + '_' + str(num_components) + '.json', 'r')
        data = json.load(f)
        f.close()
        num_frames = len(data)
        data = np.array(list(map(lambda x: [x['frame'], x['c1'], x['d1'], x['c2'], x['d2']], data)))
        order = np.argsort(data, axis=0)[:,0] # order from initial frame to last frame
        data = data[order]
        
        total_real_frames = 0
        predicted_frames_eucl = 0
        predicted_frames_mahal = 0
        for elem in real_data:
            total_real_frames += elem[2] - elem[1] + 1
            for j in range(elem[1], elem[2]+1):
                if data[j,1] == elem[0]:
                    predicted_frames_eucl += 1
                if data[j,3] == elem[0]:
                    predicted_frames_mahal += 1
        
        accuracies_eucl[i,index] = 100*float(predicted_frames_eucl)/total_real_frames
        accuracies_mahal[i,index] = 100*float(predicted_frames_mahal)/total_real_frames
        
            # =============================================================================
    # PLOT DATA AND COMPARE IT WITH REAL VALUES
    # =============================================================================
    
    colors = ['b','m','r','k','y','c','g']
    
    # Euclidean distance
    fig = plt.figure('Euclidean' + str(num_components))
    plt.suptitle('Euclidean' + str(num_components))
    
    for i, video_name in enumerate(video_names):
        # Get real labels of frames
        f2 = open(path + 'labels_real_' + video_name[:-4] + '.json', 'r')
        real_data = json.load(f2)
        f2.close()
        real_data = np.array(list(map(lambda x: [x['class'], x['frameStart'], x['frameEnd']], real_data)))
        
        # Get predicted data
        f = open(path + 'labels_' + model_name + '_' + video_name[:-4] + '_' + technique + '_' + str(num_components) + '.json', 'r')
        data = json.load(f)
        f.close()
        num_frames = len(data)
        data = np.array(list(map(lambda x: [x['frame'], x['c1'], x['d1'], x['c2'], x['d2']], data)))
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
        
    # Mahalanobis distance
    fig = plt.figure('Mahalanobis' + str(num_components))
    plt.suptitle('Mahalanobis' + str(num_components))
    
    for i, video_name in enumerate(video_names):
        # Get real labels of frames
        f2 = open(path + 'labels_real_' + video_name[:-4] + '.json', 'r')
        real_data = json.load(f2)
        f2.close()
        real_data = np.array(list(map(lambda x: [x['class'], x['frameStart'], x['frameEnd']], real_data)))
        
        # Open json file and parse data to numpy array
        f = open(path + 'labels_' + model_name + '_' + video_name[:-4] + '_' + technique + '_' + str(num_components) + '.json', 'r')
        data = json.load(f)
        f.close()
        num_frames = len(data)
        data = np.array(list(map(lambda x: [x['frame'], x['c1'], x['d1'], x['c2'], x['d2']], data)))
        order = np.argsort(data, axis=0)[:,0] # order from initial frame to last frame
        data = data[order]
        max_dist2 = data[:,4].max()
        
        # Plot predicted labels and distances to nearest neighbors
        ax1 = plt.subplot('22'+str(i+1))
        ax1.plot(range(num_frames), data[:,4], color=(.5,.5,.5,.5), linewidth=.8)
        for frame in range(num_frames):
            ax1.plot(frame, data[frame,4], colors[int(data[frame,3])] + '.')
        ax1.set_ylim(0, max_dist2)
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
if model_name == 'eigen':
    fig = plt.figure('Accuracies')
    plt.suptitle('Accuracies')
    for i in range(len(video_names)):
        ax = plt.subplot('22'+str(i+1))
    #    ax.set_title(video_names[i])
        ax.plot(num_components_array, accuracies_eucl[i], 'b', marker='o', label='Euclidean')
        ax.plot(num_components_array, accuracies_mahal[i], 'r', marker='o', label='Mahalanobis')
        plt.legend()
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Accuracy')
        plt.grid()
    plt.show()
    

    
    
