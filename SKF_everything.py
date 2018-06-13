# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:58:59 2018

@author: Pablo
"""

import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter
import cv2
import numpy as np
import os
import utils
from sklearn.neighbors import KNeighborsClassifier
import pickle
from imutils import rotate_bound
import json
import pdb.set_trace as STOP

technique = 'SKF'
neighborhood = 17
regenerate_imgs = True

video_name = 'Video3.mp4'
model_names = ['eigen', 'fisher']

def spatial_kernel_filter(array):
    range_val = np.ptp(array)
    if range_val < threshold:
        return np.mean(array)
    else:
        return range_val
    
def spatial_kernel_filter_uint8(array):
    range_val = np.ptp(array)
    if range_val < threshold:
        return int(round(np.mean(array)))
    else:
        return range_val

for threshold_real in np.arange(10, 101, 5):
    print(threshold_real)
    threshold = round(float(threshold_real) / 255, 3) # 0 < threshold < 1; example: 127 turns into 0.5
    
    # =============================================================================
    # GENERATE IMAGES WITH SPATIAL KERNEL FILTER (SKF)
    # =============================================================================
    
    if regenerate_imgs:
        path = 'C:/Users/Pablo/Google Drive/TFM/Modified Images/'
        for pathnew in os.listdir(path):
            print(pathnew)
            for fi in os.listdir(path + '../Modified Images ' + technique + '/' + pathnew):
                os.remove(path + '../Modified Images ' + technique + '/' + pathnew + '/' + fi)
            for img_name in os.listdir(path + pathnew):
                im = plt.imread(path + pathnew + '/' + img_name)
                imNew = generic_filter(im, spatial_kernel_filter, size=(neighborhood,neighborhood))
                imNew = np.stack((imNew,)*3, axis=2)
                plt.imsave(path + '../Modified Images ' + technique + '/' + pathnew + '/' + img_name, imNew)
    
    # =============================================================================
    # LOAD IMAGES AND CREATE / LOAD MODEL
    # =============================================================================
    
    path = 'C:/Users/Pablo/Google Drive/TFM/'
    
    # Load images
    class_names = os.listdir(path + 'Modified Images ' + technique + '/')
    class_numbers = list(map(lambda x: int(x[0]), class_names))
    images, labels = utils.load_images_and_labels(path + 'Modified Images ' + technique + '/')
    num_samples, height, width = images.shape
    
    for model_name in model_names:
    
        if model_name == 'eigen':
            num_components = 40
        elif model_name == 'fisher':
            num_components = 6
        else:
            print('ERROR: Wrong model_name')
            
        # Generate and save model or Load model if exists
        if not os.path.isfile(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.yml'):
            if model_name == 'eigen':
                model = cv2.face.createEigenFaceRecognizer(num_components)
            elif model_name == 'fisher':
                model = cv2.face.createFisherFaceRecognizer()
            model.train(images, labels)
#            model.save(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.yml')
        else:
            if model_name == 'eigen':
                model = cv2.face.createEigenFaceRecognizer()
            elif model_name == 'fisher':
                model = cv2.face.createFisherFaceRecognizer()
            model.load(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.yml')
        
        # Get eigenvectors, mean and projections
        eigen_imgs = model.getEigenVectors()
        mean_vector = model.getMean()
        projections = np.array(model.getProjections()).reshape(num_samples, num_components)
        classes = model.getLabels().ravel()
        
        # =============================================================================
        # CREATE / LOAD K-NN CLASSIFIERS (EUCLIDEAN AND MAHALANOBIS METRIC)
        # =============================================================================
        
        ## Generate and save classifiers or Load classifiers if exist
        if not os.path.isfile(path + 'Models/' + model_name + '_K-NN_euclidean_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.sav'):
            classif1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
            classif1.fit(projections, classes)
#            pickle.dump(classif1, open(path + 'Models/' + model_name + '_K-NN_euclidean_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.sav', 'wb'))
        else:
            classif1 = pickle.load(open(path + 'Models/' + model_name + '_K-NN_euclidean_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.sav', 'rb'))
        if not os.path.isfile(path + 'Models/' + model_name + '_K-NN_mahalanobis_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.sav'):
            classif2 = KNeighborsClassifier(n_neighbors=1, metric='mahalanobis', metric_params=dict(V=np.cov(projections, rowvar=False)))
            classif2.fit(projections, classes)
#            pickle.dump(classif2, open(path + 'Models/' + model_name + '_K-NN_mahalanobis_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.sav', 'wb'))
        else:
            classif2 = pickle.load(open(path + 'Models/' + model_name + '_K-NN_mahalanobis_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.sav', 'rb'))
            
        # =============================================================================
        # PREDICT THE LABELS FOR EACH FRAME AND SAVE IT TO A JSON FILE
        # =============================================================================
            
        path = 'C:/Users/Pablo/Google Drive/TFM/Videos/'
        
        # Open json file to store data
        f = open(path + 'labels_' + model_name + '_' + video_name[:-4] + '_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.json', 'w')
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
        else:
            transpose = False
            
        frame = 0
        
        # Store first frame's prediction
        st, im = cap.read()
        if im is not None:
            if transpose:
                im = rotate_bound(im, 90)
            im = cv2.cvtColor(cv2.resize(im, (90,160)), cv2.COLOR_BGR2GRAY)
            im = generic_filter(im, spatial_kernel_filter_uint8, size=(neighborhood,neighborhood)).astype(np.uint8).reshape(1,14400)
            im_projected = np.dot(eigen_imgs.T, (im - mean_vector).T).reshape(1,num_components)
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
            
            # Reduce size, convert to grayscale and project to subspace
            im = cv2.cvtColor(cv2.resize(im, (90,160)), cv2.COLOR_BGR2GRAY)
            im = generic_filter(im, spatial_kernel_filter_uint8, size=(neighborhood,neighborhood)).astype(np.uint8).reshape(1,14400)
            im_projected = np.dot(eigen_imgs.T, (im - mean_vector).T).reshape(1,num_components)
            
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
    # PLOT DATA AND COMPARE IT WITH REAL VALUES
    # =============================================================================
    
    # Get real labels of frames
    f2 = open(path + 'labels_real_' + video_name[:-4] + '.json', 'r')
    real_data = json.load(f2)
    f2.close()
    real_data = np.array(map(lambda x: [x['class'], x['frameStart'], x['frameEnd']], real_data))
    
    colors = ['b','m','r','k','y','c','g']
    
    # Euclidean distance
    fig = plt.figure('Euclidean Threshold ' + str(threshold_real))
    plt.suptitle(video_name[:-4] + ' | Euclidean | ' + str(threshold_real))
    
    for i in range(len(model_names)):
        # Open json file and parse data to numpy array
        if model_names[i] == 'eigen':
            num_components = 40
        else:
            num_components = 6
        f = open(path + 'labels_' + model_names[i] + '_' + video_name[:-4] + '_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.json', 'r')
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
    fig = plt.figure('Mahalanobis Threshold ' + str(threshold_real))
    plt.suptitle(video_name[:-4] + ' | Mahalanobis | ' + str(threshold_real))
    
    for i in range(len(model_names)):
        # Open json file and parse data to numpy array
        if model_names[i] == 'eigen':
            num_components = 40
        else:
            num_components = 6
        f = open(path + 'labels_' + model_names[i] + '_' + video_name[:-4] + '_' + technique + '_' + str(num_components) + '_' + str(threshold) + '.json', 'r')
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

