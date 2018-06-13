# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 12:15:19 2018

@author: Pablo
"""

from __future__ import division
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import utils

# =============================================================================
# PREPROCESS IMAGE DATASET
# =============================================================================
    
#path = 'C:/Users/Pablo/Google Drive/TFM/Images/'
## Rename image files
#utils.rename_image_files(path)
## Change image size and convert to grayscale. Save them in folder 'Modified Images'
#utils.modify_images(path)


# =============================================================================
# LOAD IMAGES AND LABELS
# =============================================================================
path = 'C:/Users/Pablo/Google Drive/TFM/Modified Images/'
class_names = os.listdir(path)
class_numbers = list(map(lambda x: int(x[0]), class_names))
images, labels = utils.load_images_and_labels(path)
height = images.shape[1]


# =============================================================================
# FISHERIMAGES: STATIC TRAINING LEAVING ONE (OF EACH CLASS) OUT
# =============================================================================

# Load test samples
test_samples = []
for i in np.arange(60, 61*len(class_numbers), 61):
    test_samples.append(images[i])
test_labels = range(len(test_samples))
images = np.delete(images, np.arange(60, 61*len(class_numbers), 61), axis=0)
labels = np.delete(labels, np.arange(60, 61*len(class_numbers), 61), axis=0)

# Train eigenimages model
model = cv2.face.createFisherFaceRecognizer()
model.train(images, labels)
fisher_imgs = model.getEigenVectors()
num_fishers = fisher_imgs.shape[1]
mean_vector = model.getMean()
projections = model.getProjections()

# Make predictions
pred = []
for i in range(len(test_samples)):
    pred.append(model.predict(test_samples[i]))
    
# Project test samples to subspace (PCA)
test_samples = np.array(map(lambda x: np.dot(fisher_imgs.T, (x.reshape(1,14400)-mean_vector).T).reshape(1, num_fishers), test_samples))

# Save some fisherimages
utils.save_fisherimages(fisher_imgs, height, path, amount=num_fishers)

# =============================================================================
# PLOT EUCLIDEAN DISTANCES
# =============================================================================

# Plot predicted class of test samples
plt.figure()
ax = plt.subplot('339')
ax.plot(class_numbers, pred, marker='o')
ax.set_xlabel('Real class')
ax.set_ylabel('Prediction')
ax.set_title('Accuracy')
ax.set_yticks(range(7))
ax.grid(True)

# Plot distances to test samples
x_values = range(len(projections))
colors = ['b','m','r','k','y','c','g']
plt.suptitle('Euclidean distances to test samples (leaving-one-out)')

for j in range(len(test_samples)):

    dists = list(map(lambda x: np.linalg.norm(test_samples[j]-x), projections))
    ax = plt.subplot('33'+str(j+1))
    ax.set_title(class_names[j][4:], fontdict={'color':colors[j]})
    plt.xticks([]) # hide x-axis values
    min_dists = []
    
    # Plot distances with different colors depending on the class
    if j == len(test_samples)-1:
        plot_handles = []
        for i in range(len(class_names)):
            plot_handles.append(plt.plot(x_values[60*i:(60*i+60)],dists[60*i:(60*i+60)], colors[i], label=class_names[i], linewidth=2)[0])
            min_dists.append(min(dists[60*i:(60*i+60)]))
    else:
        for i in range(len(class_names)):
            plt.plot(x_values[60*i:(60*i+60)],dists[60*i:(60*i+60)], colors[i], label=class_names[i][4:], linewidth=2)
            min_dists.append(min(dists[60*i:(60*i+60)]))
    
    # Plot a line with the minimum distance
    min_dist = min(dists)
    nearest_neighbour = dists.index(min_dist) // 60
    plt.plot([0,len(dists)-1], [min_dist, min_dist], colors[nearest_neighbour], linewidth=0.5)
    
    # Plot a line with the minimum distance from the second closest class
    min_dists[min_dists.index(min_dist)] = max(min_dists)
    min_dist2 = min(min_dists)
    second_class_idx = min_dists.index(min_dist2)
    plt.plot([0,len(dists)-1], [min_dist2, min_dist2], colors[second_class_idx], linewidth=0.5)
    
    # Colour the yticks
    ax.set_yticks([min_dist, min_dist2])
    [t.set_color(i) for (i,t) in zip([colors[nearest_neighbour], colors[second_class_idx]], ax.yaxis.get_ticklabels())]

# Add legend    
plt.legend(handles=plot_handles, loc=2, bbox_to_anchor=(1.4,1.05))

# Show maximized window
plt.get_current_fig_manager().window.showMaximized()

# =============================================================================
# PLOT MAHALANOBIS DISTANCES
# =============================================================================

# Plot predicted class of test samples
plt.figure()
ax = plt.subplot('339')
ax.plot(class_numbers, pred, color=(1,140./255,0), marker='o')
ax.set_xlabel('Real class')
ax.set_ylabel('Prediction')
ax.set_title('Accuracy')
ax.set_yticks(range(7))
ax.grid(True)

# Plot distances to test samples
x_values = range(len(projections))
colors = ['b','m','r','k','y','c','g']
plt.suptitle('Mahalanobis distances to test samples (leaving-one-out)')

projections = np.array(projections).reshape(len(images), num_fishers)
std_dev = np.std(projections, axis=0)

for j in range(len(test_samples)):

    dists = list(map(lambda x: np.linalg.norm((test_samples[j]-x)/std_dev), projections))
    ax = plt.subplot('33'+str(j+1))
    ax.set_title(class_names[j][4:], fontdict={'color':colors[j]})
    plt.xticks([]) # hide x-axis values
    min_dists = []
    
    # Plot distances with different colors depending on the class
    if j == len(test_samples)-1:
        plot_handles = []
        for i in range(len(class_names)):
            plot_handles.append(plt.plot(x_values[60*i:(60*i+60)],dists[60*i:(60*i+60)], colors[i], label=class_names[i], linewidth=2)[0])
            min_dists.append(min(dists[60*i:(60*i+60)]))
    else:
        for i in range(len(class_names)):
            plt.plot(x_values[60*i:(60*i+60)],dists[60*i:(60*i+60)], colors[i], label=class_names[i][4:], linewidth=2)
            min_dists.append(min(dists[60*i:(60*i+60)]))
    
    # Plot a line with the minimum distance
    min_dist = min(dists)
    nearest_neighbour = dists.index(min_dist) // 60
    plt.plot([0,len(dists)-1], [min_dist, min_dist], colors[nearest_neighbour], linewidth=0.5)
    
    # Plot a line with the minimum distance from the second closest class
    min_dists[min_dists.index(min_dist)] = max(min_dists)
    min_dist2 = min(min_dists)
    second_class_idx = min_dists.index(min_dist2)
    plt.plot([0,len(dists)-1], [min_dist2, min_dist2], colors[second_class_idx], linewidth=0.5)
    
    # Colour the yticks
    ax.set_yticks([min_dist, min_dist2])
    [t.set_color(i) for (i,t) in zip([colors[nearest_neighbour], colors[second_class_idx]], ax.yaxis.get_ticklabels())]

# Add legend
plt.legend(handles=plot_handles, loc=2, bbox_to_anchor=(1.4,1.05))

# Show maximized window
plt.get_current_fig_manager().window.showMaximized()

plt.show()
