# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 13:40:36 2018

@author: Pablo
"""

import cv2
import os
import utils
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pdb
from imutils import rotate_bound

"""
Information saved in the json file:
    frame: video frame number
    c1: predicted class by the K-NN with euclidean metric
    d1: euclidean distance to closest neighbour
    c2: predicted class by the K-NN with mahalanobis metric
    d2: mahalanobis distance to closest neighbour
"""

video_name = 'Video4.mp4'
model_name = 'eigen'

path = 'C:/Users/Pablo/Google Drive/TFM/'

if model_name == 'eigen':
    num_components = 40
elif model_name == 'fisher':
    num_components = 6
else:
    print('ERROR: Wrong model_name')

# =============================================================================
# SAVE / LOAD MODEL
# =============================================================================

# Load images
class_names = os.listdir(path + 'Modified Images/')
class_numbers = list(map(lambda x: int(x[0]), class_names))
images, labels = utils.load_images_and_labels(path + 'Modified Images/')
num_samples, height, width = images.shape

# Generate and save model
if model_name == 'eigen':
    model = cv2.face.createEigenFaceRecognizer(num_components)
elif model_name == 'fisher':
    model = cv2.face.createFisherFaceRecognizer()
model.train(images, labels)
model.save(path + 'Models/' + model_name + '.yml')

# Load model
if model_name == 'eigen':
    model = cv2.face.createEigenFaceRecognizer()
elif model_name == 'fisher':
    model = cv2.face.createFisherFaceRecognizer()
model.load(path + 'Models/' + model_name + '.yml')

eigen_imgs = model.getEigenVectors()
mean_vector = model.getMean()
projections = np.array(model.getProjections()).reshape(num_samples, num_components)
classes = model.getLabels().ravel()


# =============================================================================
# SAVE / LOAD K-NN CLASSIFIERS (EUCLIDEAN AND MAHALANOBIS METRIC)
# =============================================================================

## Generate and save classifiers
#classif1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
#classif1.fit(projections, classes)
#pickle.dump(classif1, open(path + 'Models/' + model_name + '_K-NN_euclidean.sav', 'wb'))
#
#classif2 = KNeighborsClassifier(n_neighbors=1, metric='mahalanobis', metric_params=dict(V=np.cov(projections, rowvar=False)))
#classif2.fit(projections, classes)
#pickle.dump(classif2, open(path + 'Models/' + model_name + '_K-NN_mahalanobis.sav', 'wb'))

# Load classifiers
classif1 = pickle.load(open(path + 'Models/' + model_name + '_K-NN_euclidean.sav', 'rb'))
classif2 = pickle.load(open(path + 'Models/' + model_name + '_K-NN_mahalanobis.sav', 'rb'))


# =============================================================================
# PREDICT THE LABELS FOR EACH FRAME AND SAVE IT TO A JSON FILE
# =============================================================================
    
path = 'C:/Users/Pablo/Google Drive/TFM/Videos/'

# Open json file to store data
f = open(path + 'labels_' + model_name + '_' + video_name[:-4] + '.json', 'w')
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
    im_projected = np.dot(eigen_imgs.T, (im.reshape(1,14400)-mean_vector).T).reshape(1,num_components)
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
    im_projected = np.dot(eigen_imgs.T, (im.reshape(1,14400)-mean_vector).T).reshape(1,num_components)
    
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