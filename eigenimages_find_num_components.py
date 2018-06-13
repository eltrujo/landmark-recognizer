# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:20:20 2018

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
# TEST ACCURACY WITH DIFFERENT NUMBER OF PRINCIPAL COMPONENTS
# =============================================================================
def test_accuracy_eigen(iteration):
    
    imgs_train, img_test, labels_train, label_test = train_test_split(images, labels, test_size=1) # leaving one out
#    imgs_train = np.delete(images, iteration, axis=0)
#    labels_train = np.delete(labels, iteration, axis=0)
#    img_test = images[iteration].reshape(1, height, width)
#    label_test = labels[iteration].reshape(1)
    
    # Eigenimages
    model = cv2.face.createEigenFaceRecognizer(num_components)
    model.train(imgs_train, labels_train)
    eigen_imgs = model.getEigenVectors()
    mean_vector = model.getMean()
    projections = np.array(model.getProjections()).reshape(num_samples-1, num_components)
    classes = model.getLabels().ravel()
    
    projection_test = np.dot(eigen_imgs.T, (img_test[0].reshape(1,height*width)-mean_vector).T).reshape(1,num_components)
    class_test = label_test[0]
    
    # Nearest Neighbor classifiers with euclidean and mahalanobis distances
    classif1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    classif1.fit(projections, classes)
    pred1 = classif1.predict(projection_test)[0]
    
    classif2 = KNeighborsClassifier(n_neighbors=1, metric='mahalanobis', metric_params=dict(V=np.cov(projections, rowvar=False)))
    classif2.fit(projections, classes)
    pred2 = classif2.predict(projection_test)[0]
    
    # Test models
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

# Estimate accuracy for different number of Principal Components
accuracy_vector1 = []
accuracy_vector2 = []
door_except_vector1 = []
door_except_vector2 = []
comp_range = np.arange(10, 201, 10)

for num_components in comp_range:
    
    tic=time.time()
    vals = np.array(list(map(test_accuracy_eigen, range(100))))
    print('Elapsed time: %.2f secs' % (time.time()-tic))
            
    accuracy_vector1.append(vals[:,0].sum())
    accuracy_vector2.append(vals[:,1].sum())
    door_except_vector1.append(vals[:,2].sum())
    door_except_vector2.append(vals[:,3].sum())
    print('Num components:' + str(num_components))

# Normalize accuracy vectors
acc_vector1 = list(map(lambda x: x/len(images), accuracy_vector1))
acc_vector2 = list(map(lambda x: x/len(images), accuracy_vector2))

# =============================================================================
# PLOT RESULTS
# =============================================================================
fig = plt.figure()
ax = fig.add_subplot(111)
plt.suptitle('Nearest Neighbor with euclidean and mahalanobis metrics\n(leaving-one-out)')
plot1 = plt.plot(comp_range, acc_vector1, 'b', marker='o', label='Euclidean')[0]
plot2 = plt.plot(comp_range, acc_vector2, 'r', marker='o', label='Mahalanobis')[0]
plt.legend(handles=[plot1, plot2])
plt.xlabel('Number of Principal Components')
plt.ylabel('Accuracy')
ax.axis([comp_range[0]-5, comp_range[-1]+5, .94, 1.005])
ax.set_yticklabels(['%.i%%' % (x*100) for x in ax.get_yticks()])
plt.grid(True)

if max(acc_vector1) > max(acc_vector2):
    max_acc = max(acc_vector1)
    max_col = 'b'
else:
    max_acc = max(acc_vector2)
    max_col = 'r'
ax2 = ax.twinx()
ax2.plot([comp_range[0]-5, comp_range[-1]+5], [max_acc, max_acc], max_col, linewidth=0.5)
ax2.set_yticks([max_acc])
ax2.set_yticklabels(['%.2f%%' % (x*100) for x in ax2.get_yticks()])
ax2.axis([comp_range[0]-5, comp_range[-1]+5, .94, 1.005])

#plt.get_current_fig_manager().window.showMaximized()
#fig.savefig('Accuracy vs Num_components.png')

plt.show()

# Go to sleep
#check_output('rundll32.exe powrprof.dll,SetSuspendState 0,1,0', shell=True)
