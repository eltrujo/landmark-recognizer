# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:33:17 2018

@author: Pablo
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle
import utils
from sklearn.neighbors import KNeighborsClassifier
from imutils import rotate_bound
import json
from pdb import set_trace as STOP

technique = 'orb_align'
video_names = ['Video1.mp4', 'Video2.mp4', 'Video3.mp4', 'Video4.mp4']
model_name = 'eigen'
#model_name = 'fisher'
MAX_MATCHES = 200
GOOD_MATCH_PERCENT = 0.15
save_comp = False

if model_name == 'eigen':
    num_components_array = np.arange(40, 41, 1) # for eigen
elif model_name == 'fisher':
    num_components_array = np.arange(6,7,1) # for fisher

if save_comp:
    img_blank = np.zeros((1280,720), dtype=np.uint8)
    def build_figure():
        plt.figure(1)
        ax1 = plt.subplot('131')
        ax1.set_title('Original frame')
        ax1.axis('off')
        ax2 = plt.subplot('132')
        ax2.set_title('Aligned frame')
        ax2.axis('off')
        ax3 = plt.subplot('133')
        ax3.set_title('Reference image for alignment')
        ax3.axis('off')
        return ax1, ax2, ax3


def extract_features_ORB(images, orb):
    descriptors = []
    keypoints = []
    for img in images:
        keyp, descript = orb.detectAndCompute(img, None)
        keypoints.append(keyp)
        descriptors.append(descript)
    return keypoints, descriptors

def align_image(im):
    keypts, desc = extract_features_ORB([im], orb)
    keypts = keypts[0]
    desc = desc[0]
    if desc is not None and len(desc) > MAX_MATCHES*0.75:
        ims_aligned = []
        for j, elem in enumerate(descriptors):
            matches = matcher.match(desc, elem, None) # match features
            matches.sort(key=lambda x: x.distance, reverse=False) # sort matches by score
            matches = matches[:int(len(matches)*GOOD_MATCH_PERCENT)] # remove not so good matches
            # Extract location of good matches
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)
            for i, match in enumerate(matches):
                points1[i, :] = keypts[match.queryIdx].pt
                points2[i, :] = keypoints[j][match.trainIdx].pt
            # Find homography
            hom, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            if hom is not None:
                ims_aligned.append(cv2.warpPerspective(im, hom, (w, h)))
            else:
                ims_aligned.append(im)
        return ims_aligned
    else:
        return None

#def get_closest_neighbor(ims_aligned):    
#    datMat = np.zeros((len(ims_aligned), 4))
#    for j, im_aligned in enumerate(ims_aligned):
#        im = cv2.resize(im_aligned, (90,160))
#        im_projected = np.dot(eigen_imgs.T, (im.reshape(1,14400)-mean_vector).T).reshape(1,num_components)
#        # Get nearest neighbor (euclidean and mahalanobis metrics)
#        dist1, ind1 = classif1.kneighbors(X=im_projected, n_neighbors=1, return_distance=True)
#        datMat[j,0] = classes[ind1[0][0]]
#        datMat[j,1] = round(dist1[0][0])
#        dist2, ind2 = classif2.kneighbors(X=im_projected, n_neighbors=1, return_distance=True)
#        datMat[j,2] = classes[ind2[0][0]]
#        datMat[j,3] = round(dist2[0][0],3)
#    ind1 = np.argmin(datMat[:,1])
#    ind2 = np.argmin(datMat[:,3])
#    return [datMat[ind1,0], datMat[ind1,1], datMat[ind2,2], datMat[ind2,3]]
        
def get_closest_neighbor_hist(ims_aligned, im):
    hists = np.zeros((len(ims_aligned),256), dtype=np.int64)
    for j, im_aligned in enumerate(ims_aligned):
        hists[j,:] = np.bincount(im_aligned.ravel(),minlength=256)
    hists = hists - np.bincount(im.ravel(),minlength=256)
    ind = np.argmin(np.linalg.norm(hists,axis=1))
    img = cv2.resize(ims_aligned[ind], (90,160))
    im_projected = np.dot(eigen_imgs.T, (img.reshape(1,14400)-mean_vector).T).reshape(1,num_components)
    # Get nearest neighbor (euclidean and mahalanobis metrics)
    dist1, ind1 = classif1.kneighbors(X=im_projected, n_neighbors=1, return_distance=True)
    class1 = classes[ind1[0][0]]
    dist2, ind2 = classif2.kneighbors(X=im_projected, n_neighbors=1, return_distance=True)
    class2 = classes[ind2[0][0]]
    [class1, round(dist1[0][0]), class2, round(dist2[0][0],3)]
    return [class1, round(dist1[0][0]), class2, round(dist2[0][0],3)]

# =============================================================================
# LOAD IMAGES
# =============================================================================
path = 'C:/Users/Pablo/Google Drive/TFM/'

# Load images normal size
class_names = os.listdir(path + 'Images/')
class_numbers = list(map(lambda x: int(x[0]), class_names))
images, labels = utils.load_images_and_labels(path + 'Images/')
num_samples, height, width = images.shape
images_mean = []
images_closest = []
for i in np.arange(0,61*6+1,61):
    images_mean.append(np.mean(images[i:i+61],axis=0))
    index = i + np.argmin(np.sum(np.abs(images[i:i+61] - images_mean[-1]),axis=(1,2)))
    images_closest.append(images[index])
if save_comp:
    ims_closest = []
    for im in images_closest:
        ims_closest.append(cv2.cvtColor(im,cv2.COLOR_GRAY2BGR))
    ims_closest.append(cv2.cvtColor(img_blank, cv2.COLOR_GRAY2BGR))

# Load images reduced
class_names = os.listdir(path + 'Modified Images/')
class_numbers = list(map(lambda x: int(x[0]), class_names))
images, labels = utils.load_images_and_labels(path + 'Modified Images/')
num_samples, height, width = images.shape
    
# =============================================================================
# DETECT ORB FEATURES OF ONE IMAGE FOR EACH LANDMARK
# =============================================================================
orb = cv2.ORB_create(MAX_MATCHES)
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
keypoints, descriptors = extract_features_ORB(images_closest, orb) # each image has MAX_MATCHES descriptors with 32 items each
    

accuracies_eucl = np.zeros((len(video_names),len(num_components_array)), dtype=np.float)
accuracies_mahal = np.zeros((len(video_names),len(num_components_array)), dtype=np.float)
    
for index, num_components in enumerate(num_components_array):
    
    path = 'C:/Users/Pablo/Google Drive/TFM/'
    
    # =============================================================================
    # CREATE/LOAD RECOGNIZER
    # =============================================================================
    if not os.path.isfile(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml'):
        if model_name =='eigen':
            model = cv2.face.EigenFaceRecognizer_create(num_components)
        elif model_name == 'fisher':
            model = cv2.face.FisherFaceRecognizer_create()
        model.train(images, labels)
        model.write(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml')
    else:
        if model_name =='eigen':
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
        
        if save_comp:
            fig = plt.figure(1)
            fig.clf()
            fig.set_size_inches(10,7)
            ax_frame, ax_aligned, ax_ref = build_figure()
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(path + video_name[:-4] + '_' + technique + '_' + str(num_components) + '.avi', fourcc, 30.0, (1000,700), True)
        
        # Store first frame's prediction
        st, im = cap.read()
        if im is not None:
            if transpose:
                im = rotate_bound(im, 90)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ims_aligned = align_image(im)
            if ims_aligned is not None:
#                data_values = get_closest_neighbor(ims_aligned)
                data_values = get_closest_neighbor_hist(ims_aligned, im)
            else:
                # If not enough features were detected, assign class 7 (white color) and distance 0
                data_values = [7,0,7,0]
                ims_aligned.append(img_blank)
            if save_comp:
                ax_frame.imshow(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
                ax_aligned.imshow(cv2.cvtColor(ims_aligned[int(data_values[0])], cv2.COLOR_GRAY2RGB))
                ax_ref.imshow(ims_closest[int(data_values[0])])
                # Get figure as image and save it to video
                img = utils.fig2data(fig)
                out.write(img)
                
            frame += 1
                
            # Store data into json file
            info = [frame, data_values[0], data_values[1], data_values[2], data_values[3]]
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
            ims_aligned = align_image(im)
            if ims_aligned is not None:
#                data_values = get_closest_neighbor(ims_aligned)
                data_values = get_closest_neighbor_hist(ims_aligned, im)
            else:
                # If not enough features were detected, assign class 7 (white color) and distance 0
                data_values = [7,0,7,0]
            if save_comp:
                ax_frame.imshow(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
                if ims_aligned is not None:
                    ax_aligned.imshow(cv2.cvtColor(ims_aligned[int(data_values[0])], cv2.COLOR_GRAY2RGB))
                else:
                    ax_aligned.imshow(ims_closest[int(data_values[0])])
                ax_ref.imshow(ims_closest[int(data_values[0])])
                # Get figure as image and save it to video
                img = utils.fig2data(fig)
                out.write(img)
                if frame % 36 == 0:
                    fig.clf()
                    ax_frame, ax_aligned, ax_ref = build_figure()
            # Store data into json file
            info = [frame, data_values[0], data_values[1],  data_values[2],  data_values[3]]
            f.write(',\n' + str(dict(zip(keys, info))).replace("'", '"'))
            frame += 1
        
        f.write('\n]')
        f.close()
        cap.release()
        if save_comp:
            out.release()
    
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
    
    colors = ['b','m','r','k','y','c','g','w']
    
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
    fig = plt.figure()
    plt.get_current_fig_manager().window.showMaximized()
    plt.suptitle('Accuracies')
    ax = []
    for i in range(len(video_names)):
        ax.append( plt.subplot('22'+str(i+1)))
        ax[-1].plot(num_components_array, accuracies_eucl[i], 'b', marker='o', label='Euclidean')
        ax[-1].plot(num_components_array, accuracies_mahal[i], 'r', marker='o', label='Mahalanobis')
        plt.xlabel('Num components')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
    plt.show()

#    for i in range(4):
#        ax[i].set_yticklabels(['%.0f%%' % (x) for x in ax[i].get_yticks()])