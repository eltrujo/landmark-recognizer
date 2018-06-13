# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 17:20:52 2018

@author: Pablo
"""

from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from subprocess import check_output
import time


def test_accuracy_fisher(iteration):
    print(iteration)
    
    # Split training samples from testing sample
    imgs_train = np.delete(images, iteration, axis=0)
    labels_train = np.delete(labels, iteration, axis=0)
    img_test = images[iteration].reshape(1, height, width)
    label_test = labels[iteration].reshape(1)
    
    # Fisherimages
    model = cv2.face.createFisherFaceRecognizer()
    model.train(imgs_train, labels_train)
    fisher_imgs = model.getEigenVectors()
    mean_vector = model.getMean()
    projections = np.array(model.getProjections()).reshape(num_samples-1, num_fishers)
    classes = model.getLabels().ravel()
    
    # Project test sample into subspace and retrieve its class
    projection_test = np.dot(fisher_imgs.T, (img_test[0].reshape(1,height*width)-mean_vector).T).reshape(1,num_fishers)
    class_test = label_test[0]
    
    # Nearest Neighbor classifiers with euclidean and mahalanobis distances
    classif1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    classif1.fit(projections, classes)
    pred1 = classif1.predict(projection_test)[0]
    
    classif2 = KNeighborsClassifier(n_neighbors=1, metric='mahalanobis', metric_params=dict(V=np.cov(projections, rowvar=False)))
    classif2.fit(projections, classes)
    pred2 = classif2.predict(projection_test)[0]
    
    # Test if prediction was right
    accuracy1 = 0
    accuracy2 = 0
    door_exception1 = 0
    door_exception2 = 0
    if pred1 == class_test:
        accuracy1 = 1
    elif (pred1==2 and class_test==3) or (pred1==3 and class_test==2):
        door_exception1 = 1
    if pred2 == class_test:
        accuracy2 = 1         
    elif (pred2==2 and class_test==3) or (pred2==3 and class_test==2):
        door_exception2 = 1
    return np.array([accuracy1, accuracy2, door_exception1, door_exception2], dtype=np.int)

# =============================================================================
# LOAD IMAGES AND LABELS
# =============================================================================
path = 'C:/Users/Pablo/Google Drive/TFM/Modified Images/'
images = []
labels = []
class_names = os.listdir(path)
class_numbers = list(map(lambda x: int(x[0]), class_names))
class_dirs = list(map(lambda x: path + x + '/', class_names))
for i in range(len(class_numbers)):
    img_names = os.listdir(class_dirs[i])
    for j in range(len(img_names)):
        images.append(cv2.cvtColor(cv2.imread(class_dirs[i] + img_names[j]), cv2.COLOR_BGR2GRAY))
        labels.append(class_numbers[i])
images = np.array(images)
labels = np.array(labels, dtype=np.int)
num_samples, height, width = images.shape


# =============================================================================
# TEST FISHERIMAGES ACCURACY. CROSS-VALIDATION LEAVING ONE OUT
# =============================================================================

num_fishers = len(class_numbers)-1

tic=time.time()
vals = np.array(list(map(test_accuracy_fisher, range(len(images)))))
print('Elapsed time: %.2f secs' % (time.time()-tic))
        
accuracy_1 = vals[:,0].sum() / len(images)
accuracy_2 = vals[:,1].sum() / len(images)
door_except1 = vals[:,2].sum()
door_except2 = vals[:,3].sum()

print('1NN Fisherfaces accuracy (measured with cross-validation leaving one out):')
print('Euclidean distance: %.2f %%' % (accuracy_1*100))
print('Mahalanobis distance: %.2f %%' % (accuracy_2*100))
