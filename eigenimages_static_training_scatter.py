# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 14:43:49 2018

@author: Pablo
"""
from __future__ import division
import numpy as np
import cv2
import os
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
import utils


def compute_minimum_distances(num_components):
    """
    num_components: number of Principal Components for eigenimages
    num_components type: int
        Compute a levaing one (of each class) out 61 times and calculate distance to
        closest neighbour and distance to closest neighbour of second closest class
    """
    vector_eucl = np.zeros((len(class_numbers), 4, 61))
    vector_mahal = np.zeros((len(class_numbers), 4, 61))
    
    for k in range(61):
        print(str(k))
        # Load test samples
        test_samples = []
        for i in np.arange(k, 61*len(class_numbers), 61):
            test_samples.append(images[i])
        images_train = np.delete(images, np.arange(k, 61*len(class_numbers), 61), axis=0)
        labels_train = np.delete(labels, np.arange(k, 61*len(class_numbers), 61), axis=0)
        
        # Train eigenimages model
        model = cv2.face.createEigenFaceRecognizer(num_components)
        model.train(images_train, labels_train)
        eigen_imgs = model.getEigenVectors()
        #map(lambda x: model.setLabelInfo(int(x[0]), x[4:]), class_names) # define each class's name
        mean_vector = model.getMean()
        projections = model.getProjections()
        projections = np.array(projections).reshape(len(projections), num_components)
        
        # Standard deviation of each variable, for mahalanobis metric
        std_dev = np.std(projections, axis=0)
            
        # Project test samples to subspace (PCA)
        test_samples = np.array(map(lambda x: np.dot(eigen_imgs.T, (x.reshape(1,14400)-mean_vector).T).reshape(1,num_components), test_samples))
        
        for i in range(len(class_numbers)):
            dist_eucl = np.array(map(lambda x: np.linalg.norm(test_samples[i]-x), projections))
            dist_mahal = np.array(map(lambda x: np.linalg.norm((test_samples[i]-x)/std_dev), projections))
            min_dists_eucl = []
            min_dists_mahal = []
            for j in range(len(class_numbers)):
                min_dists_eucl.append(min(dist_eucl[60*j:(60*j+60)]))
                min_dists_mahal.append(min(dist_mahal[60*j:(60*j+60)]))
            
            # Compute closest neighbour distance and second closest class's distance
            min_dist_eucl = min(min_dists_eucl)
            first_idx_eucl = min_dists_eucl.index(min_dist_eucl)
            min_dists_eucl[first_idx_eucl] = max(min_dists_eucl)
            min_dist_eucl2 = min(min_dists_eucl)
            second_idx_eucl = min_dists_eucl.index(min_dist_eucl2)
            
            min_dist_mahal = min(min_dists_mahal)
            first_idx_mahal = min_dists_mahal.index(min_dist_mahal)
            min_dists_mahal[first_idx_mahal] = max(min_dists_mahal)
            min_dist_mahal2 = min(min_dists_mahal)
            second_idx_mahal = min_dists_mahal.index(min_dist_mahal2)
            
            vector_eucl[i,0,k] = min_dist_eucl
            vector_eucl[i,1,k] = first_idx_eucl
            vector_eucl[i,2,k] = min_dist_eucl2
            vector_eucl[i,3,k] = second_idx_eucl
            
            vector_mahal[i,0,k] = min_dist_mahal
            vector_mahal[i,1,k] = first_idx_mahal
            vector_mahal[i,2,k] = min_dist_mahal2
            vector_mahal[i,3,k] = second_idx_mahal
    
    return vector_eucl, vector_mahal

def add_letter(idx, val):
    """
    idx: index of the current ytick
    idx type: int
    val: value of the current ytick
    val type: float or int
        This function is used to edit the labels of the y axis (yticks)
    """
    if idx==0:
        return 'r =\n'+str(int(round(val)))
    else:
        return 'R =\n'+str(int(round(val)))
    
def add_letter_float(idx, val):
    """
    idx: index of the current ytick
    idx type: int
    val: value of the current ytick
    val type: float
        This function is used to edit the labels of the y axis (yticks)
    """
    if idx==0:
        return 'r =\n'+str(round(val*100)/100)
    else:
        return 'R =\n'+str(round(val*100)/100)
    

# =============================================================================
# LOAD IMAGES AND LABELS
# =============================================================================

path = 'C:/Users/Pablo/Google Drive/TFM/Modified Images/'
class_names = os.listdir(path)
class_numbers = list(map(lambda x: int(x[0]), class_names))
images, labels = utils.load_images_and_labels(path)
height = images.shape[1]


# =============================================================================
# EIGENIMAGES: STATIC TRAINING LEAVING ONE (OF EACH CLASS) OUT
# =============================================================================

vector_eucl, vector_mahal = compute_minimum_distances(40)


# =============================================================================
# SCATTER EUCLIDEAN DISTANCES
# =============================================================================

plt.figure()
colors = ['b','m','r','k','y','c','g']
plt.suptitle('Minimum euclidean distances\n(r: average distance to closest neighbour; R: average distance to closest neighbour of second closest class)')
plot_handles = []

for i in range(len(class_numbers)):
    ax = plt.subplot('42'+str(i+1))
    ax.set_title(class_names[i][4:], fontdict={'color':colors[i]})
    plt.xticks([]) # hide x-axis values
    
    # Create random angles for a two-dimensional scattering plot
    rand_angles = np.random.rand(61) * 2 * np.pi
    x_vals = vector_eucl[i,0,:] * np.cos(rand_angles)
    y_vals = vector_eucl[i,0,:] * np.sin(rand_angles)
    rand_angles = np.random.rand(61) * 2 * np.pi
    x_vals2 = vector_eucl[i,2,:] * np.cos(rand_angles)
    y_vals2 = vector_eucl[i,2,:] * np.sin(rand_angles)
    save_handle = True
    for j in range(61):
        if save_handle and int(vector_eucl[i,1,j]) == i:
            plot_handles.append(plt.plot(x_vals[j], y_vals[j], color=colors[int(vector_eucl[i,1,j])], marker='.', markerSize=2.3, label=class_names[i])[0])
            save_handle = False
        else:
            plt.plot(x_vals[j], y_vals[j], color=colors[int(vector_eucl[i,1,j])], marker='.', markerSize=2.3)
        plt.plot(x_vals2[j], y_vals2[j], color=colors[int(vector_eucl[i,3,j])], marker='.', markerSize=2.3)
    
    # Plot circles with average distances
    r1 = np.mean(vector_eucl[i,0,:])
    r2 = np.mean(vector_eucl[i,2,:])
    col1 = colors[np.argmax(np.bincount(vector_eucl[i,1].astype(np.int)))]
    col2 = colors[np.argmax(np.bincount(vector_eucl[i,3].astype(np.int)))]
    
    utils.draw_circle(ax, [0,0], r1, col1)
    utils.draw_circle(ax, [0,0], r2, col2)
    
    # Draw circle and toroid
    w = Wedge((0, 0), vector_eucl[i,0].max(), 0, 360, fc=col1, ec=None, alpha=0.2)
    ax.add_artist(w)
    
    limit_val = vector_eucl[i,2].max()
    w = Wedge((0, 0), limit_val, 0, 360, width=limit_val-vector_eucl[i,2].min(), fc=col2, ec=None, alpha=0.2)
    ax.add_artist(w)
    
    # Set and colour the yticks
    plt.plot([-limit_val, 0], [r1, r1], color=col1, lineWidth=0.4, alpha=0.3)
    plt.plot([-limit_val, 0], [r2, r2], color=col2, lineWidth=0.4, alpha=0.3)
    ax.set_yticks([r1, r2])
    ax.set_yticklabels([add_letter(idx, x) for idx, x in enumerate(ax.get_yticks())])
    [t.set_color(i) for (i,t) in zip([col1, col2], ax.yaxis.get_ticklabels())]
#    
#    ax.axis('equal')
#    ax.set_autoscale_on(False)
    ax.axis([-limit_val, limit_val, -limit_val, limit_val])

# Add legend
plt.legend(handles=plot_handles, loc=2, bbox_to_anchor=(1.4,.9))

# Show maximized window
plt.get_current_fig_manager().window.showMaximized()


# =============================================================================
# SCATTER MAHALANOBIS DISTANCES
# =============================================================================

# Plot predicted class of test samples
plt.figure()
colors = ['b','m','r','k','y','c','g']
plt.suptitle('Minimum mahalanobis distances\n(r: average distance to closest neighbour; R: average distance to closest neighbour of second closest class)')
plot_handles = []

for i in range(len(class_numbers)):
    ax = plt.subplot('24'+str(i+1))
    ax.set_title(class_names[i][4:], fontdict={'color':colors[i]})
    plt.xticks([]) # hide x-axis values
    
    # Create random angles for a two-dimensional scattering plot
    rand_angles = np.random.rand(61) * 2 * np.pi
    x_vals = vector_mahal[i,0,:] * np.cos(rand_angles)
    y_vals = vector_mahal[i,0,:] * np.sin(rand_angles)
    rand_angles = np.random.rand(61) * 2 * np.pi
    x_vals2 = vector_mahal[i,2,:] * np.cos(rand_angles)
    y_vals2 = vector_mahal[i,2,:] * np.sin(rand_angles)
    save_handle = True
    for j in range(61):
        if save_handle and int(vector_mahal[i,1,j]) == i:
            plot_handles.append(plt.plot(x_vals[j], y_vals[j], color=colors[int(vector_mahal[i,1,j])], marker='.', markerSize=2.3, label=class_names[i])[0])
            save_handle = False
        else:
            plt.plot(x_vals[j], y_vals[j], color=colors[int(vector_mahal[i,1,j])], marker='.', markerSize=2.3)
        plt.plot(x_vals2[j], y_vals2[j], color=colors[int(vector_mahal[i,3,j])], marker='.', markerSize=2.3)
    
    # Plot circles with average distances
    r1 = np.mean(vector_mahal[i,0,:])
    r2 = np.mean(vector_mahal[i,2,:])
    col1 = colors[np.argmax(np.bincount(vector_mahal[i,1].astype(np.int)))]
    col2 = colors[np.argmax(np.bincount(vector_mahal[i,3].astype(np.int)))]
    
    utils.draw_circle(ax, [0,0], r1, col1)
    utils.draw_circle(ax, [0,0], r2, col2)
    
    # Draw circle and toroid
    w = Wedge((0, 0), vector_mahal[i,0].max(), 0, 360, fc=col1, ec=None, alpha=0.2)
    ax.add_artist(w)
    
    limit_val = vector_mahal[i,2].max()
    w = Wedge((0, 0), limit_val, 0, 360, width=limit_val-vector_mahal[i,2].min(), fc=col2, ec=None, alpha=0.2)
    ax.add_artist(w)
    
    # Set and colour the yticks
    plt.plot([-limit_val, 0], [r1, r1], color=col1, lineWidth=0.4)
    plt.plot([-limit_val, 0], [r2, r2], color=col2, lineWidth=0.4)
    ax.set_yticks([r1, r2])
    ax.set_yticklabels([add_letter_float(idx, x) for idx, x in enumerate(ax.get_yticks())])
    [t.set_color(i) for (i,t) in zip([col1, col2], ax.yaxis.get_ticklabels())]
    
    ax.axis([-limit_val, limit_val, -limit_val, limit_val])

# Add legend
plt.legend(handles=plot_handles, loc=2, bbox_to_anchor=(1.4,.9))

# Show maximized window
plt.get_current_fig_manager().window.showMaximized()

plt.show()