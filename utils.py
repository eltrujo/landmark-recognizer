# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:53:39 2018

@author: Pablo
"""
import os
import re
import numpy as np
import cv2
import matplotlib.pyplot
from PIL import Image

def rename_image_files(path):
    """
    path: folder that contains subfolders with the name of the class of the images inside each of them
    path type: string
        Each image will be renamed to '1.png', '2.png', ... inside each subfolder
    """
    all_dirs = map(lambda x: path + x + '/', os.listdir(path))
    
    def change_names(current_dir):
        img_names = os.listdir(current_dir)
        for i in range(len(img_names)):
            os.rename(current_dir + img_names[i], current_dir + str(i+1) + '.png')
    
    map(change_names, all_dirs)
    print('Image files renamed')
    
def modify_images(path):
    """
    path: folder (Images) that contains subfolders with the name of the class of the images inside each of them
    path type: string
        (Change image size), (convert to grayscale) and save them in folder Modified Images or Reduced Images
    """
    save_path = path + '../Modified Images/'
#    save_path = path + '../Modified Images Gray/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_dir_names = os.listdir(path)
    for current_dir_name in all_dir_names:
        if not os.path.exists(save_path + current_dir_name):
            os.makedirs(save_path + current_dir_name)
        img_names = os.listdir(path + current_dir_name)
        for i in range(len(img_names)):
            # Reduce size
#            im = cv2.resize(cv2.imread(path + current_dir_name + '/' + img_names[i]), (90,160))
            # Reduce size and convert to grayscale
            im = cv2.cvtColor(cv2.resize(cv2.imread(path + current_dir_name + '/' + img_names[i]), (90,160)), cv2.COLOR_BGR2GRAY)
            # Convert to grayscale
#            im = cv2.cvtColor(cv2.imread(path + current_dir_name + '/' + img_names[i]), cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path + current_dir_name + '/' + str(i+1) + '.png', im)
    
    print('Images modified')
    
def norm_and_reshape(vector, h):
    """
    vector: flattened eigenimage
    vector type: 1-dimension numpy array
        Normalize values to range 0-255 and reshape to original image aspect 
    """
    vector = vector - vector.min()
    vector = 255 * vector / vector.max()
    vector = np.array(list(map(round, vector)), dtype=np.uint8)
    im = vector.reshape((h, vector.shape[0]/h))
    return im

def save_eigenimages(eigen_imgs, h, path, amount=5):
    """
    eigen_imgs: flattened eigen vectors
    eigen_imgs type: 2-dimension numpy array
    h: height of the original images which the flattened arrays will be reshaped to
    h type: int
    path: folder containing image dataset ('Modified Images')
    path type: string
    amount: number of eigenimages that will be saved in folder 'Eigenimages'
    amount type: int
    """
    for i in range(amount):
        # Normalize (0 to 255) and reshape
        gray_im = norm_and_reshape(eigen_imgs[:,i], h)
        # Apply colormap
        color_im = cv2.applyColorMap(gray_im, cv2.COLORMAP_OCEAN)
        cv2.imwrite(path + '../Eigenimages/eig_' + str(i+1) + '.png', color_im)

def save_fisherimages(eigen_imgs, h, path, amount=6):
    """
    eigen_imgs: flattened eigen vectors
    eigen_imgs type: 2-dimension numpy array
    h: height of the original images which the flattened arrays will be reshaped to
    h type: int
    path: folder containing image dataset ('Modified Images')
    path type: string
    amount: number of eigenimages that will be saved in folder 'Eigenimages'
    amount type: int
    """
    for i in range(amount):
        # Normalize (0 to 255) and reshape
        gray_im = norm_and_reshape(eigen_imgs[:,i], h)
        # Apply colormap
        color_im = cv2.applyColorMap(gray_im, cv2.COLORMAP_OCEAN)
        cv2.imwrite(path + '../Fisherimages/fish_' + str(i+1) + '.png', color_im)
        
def load_images_and_labels(path):
    """
    path: folder containing image dataset
    path type: string
        Loads images and converts them to grayscale.
    """
    images = []
    labels = []
    class_names = os.listdir(path)
    class_numbers = list(map(lambda x: int(x[0]), class_names))
    class_dirs = list(map(lambda x: path + x + '/', class_names))
    for i in range(len(class_numbers)):
        img_names = os.listdir(class_dirs[i])
        img_names = sorted(img_names, key=lambda x: (int(re.sub('\D','',x)),x)) # order alphabetically
        for j in range(len(img_names)):
            images.append(cv2.cvtColor(cv2.imread(class_dirs[i] + img_names[j]), cv2.COLOR_BGR2GRAY))
            labels.append(class_numbers[i])
    images = np.array(images)
    labels = np.array(labels, dtype=np.int)
    return images, labels

def load_imagesRGB_and_labels(path):
    """
    path: folder containing image dataset ('Modified Images Reduced')
    path type: string
    """
    images = []
    labels = []
    class_names = os.listdir(path)
    class_numbers = list(map(lambda x: int(x[0]), class_names))
    class_dirs = list(map(lambda x: path + x + '/', class_names))
    for i in range(len(class_numbers)):
        img_names = os.listdir(class_dirs[i])
        img_names = sorted(img_names, key=lambda x: (int(re.sub('\D','',x)),x)) # order alphabetically
        for j in range(len(img_names)):
            images.append(cv2.imread(class_dirs[i] + img_names[j]))
            labels.append(class_numbers[i])
    images = np.array(images)
    labels = np.array(labels, dtype=np.int)
    return images, labels

def load_images_and_labels_2d(path):
    """
    path: folder containing image dataset ('Modified Images 2D')
    path type: string
    """
    images = []
    labels = []
    class_names = os.listdir(path)
    class_numbers = list(map(lambda x: int(x[0]), class_names))
    class_dirs = list(map(lambda x: path + x + '/', class_names))
    for i in range(len(class_numbers)):
        img_names = os.listdir(class_dirs[i])
        img_names = sorted(img_names, key=lambda x: (int(re.sub('\D','',x)),x)) # order alphabetically
        for j in range(len(img_names)):
            # Flatten two layers of the image
            images.append(cv2.imread(class_dirs[i] + img_names[j])[:,:,:-1].flatten().reshape(1,160*90*2))
            labels.append(class_numbers[i])
    images = np.array(images)
    labels = np.array(labels, dtype=np.int)
    return images, labels

def draw_circumference(ax, c, radius, col):
    """
    ax: axes where the circle will be plotted
    ax type: matplotlib axes
    c: x and y coordinates of the center of the circle
    c type: 2-element list or array
    col: color of the circle
    col type: any matplotlib's allowed color format
    """
    angle = np.arange(0, 2*np.pi, 0.01)
    x = c[0] + radius * np.cos(angle)
    y = c[1] + radius * np.sin(angle)
    ax.plot(x, y, color=col, lineWidth=1, alpha=0.3)
    
def choose_class(num):
    if num == 0:
        return 'Classroom'
    elif num == 1:
        return 'Corridor'
    elif num == 2:
        return 'DoorClose'
    elif num == 3:
        return 'DoorOpen'
    elif num == 4:
        return 'Elevator'
    elif num == 5:
        return 'HallBack'
    elif num == 6:
        return 'HallFront'
    
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA or RGB values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()

#    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    
    buf.shape = (h, w, 3)
 
#    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#    buf = np.roll (buf, 3, axis=2)
    return buf

def get_gif_frames(gif_name):
    frame = Image.open(gif_name)
    im = np.fromstring(frame.tobytes(), dtype=np.uint8)
    im = im.reshape((frame.size[1], frame.size[0], 3))
    
    
def modify_video(path, video_name):
    """
    path: folder path (ending with a slash: /) containing video file
    path type: str
    video_name: name of the video (including its extension)
    video_name type: str
        Convert video to grayscale and adjust height and width to 160 and 90 respectively
    """
#    save_path_name = path + '../Modified Videos/' + video_name[:-4] + '.avi'
    save_path_name = path + '../Modified Videos/' + 'Video prueba2' + '.avi'
    cap = cv2.VideoCapture(path + video_name)
    if not cap.isOpened():
        print('Error opening video file')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(save_path_name, fourcc, 30.0, (90,160), True)
        while(True):
            st, im = cap.read()
            if im is None:
                break
            im = cv2.cvtColor(cv2.resize(im, (90,160)), cv2.COLOR_BGR2GRAY)
            out.write(im)
            cv2.imshow('Hola', im)
            cv2.waitKey(1)
        cap.release()
        out.release()
        
    
#video_path = 'C:/Users/Pablo/Google Drive/TFM/Videos/'
#video = 'Video 1.mp4'
#modify_video(video_path, video)
        



    