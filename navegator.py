# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:22:20 2018

@author: Pablo
"""
import os
from glob2 import glob
from sklearn.cluster import KMeans
import json
from imutils import rotate_bound
import cv2
import numpy as np
from matplotlib import pyplot as plt
from re import sub as re_sub
import pickle
from pdb import set_trace as STOP


model_name = 'eigen'
technique = 'nav'
video_names = ['Video1.mp4']
REFS_PER_LAND = 5
MAX_MATCHES = 300
GOOD_MATCH_PERCENT = 0.05
landmarks = {"Classroom": [0,None],
             "Corridor": [1,[2,3,0]],
             "DoorClose": [2,None],
             "DoorOpen": [3,[0]],
             "Elevator": [4,[5,6]],
             "HallBack": [5,[6,1]],
             "HallFront": [6,[1,2,3]]}
current_land = "Elevator"
future_lands = landmarks[current_land][1]


def load_images_and_labels(path):
    images = []
    labels = []
    class_names = os.listdir(path)
    class_numbers = list(map(lambda x: int(x[0]), class_names))
    class_dirs = list(map(lambda x: path + x + '/', class_names))
    for i in range(len(class_numbers)):
        img_names = os.listdir(class_dirs[i])
        img_names = sorted(img_names, key=lambda x: (int(re_sub('\D','',x)),x)) # order alphabetically
        for j in range(len(img_names)):
            images.append(cv2.cvtColor(cv2.imread(class_dirs[i] + img_names[j]), cv2.COLOR_BGR2GRAY))
            labels.append(class_numbers[i])
    images = np.array(images)
    labels = np.array(labels, dtype=np.int)
    return images, labels

def load_ref_images(num_clusters=4):
    path = 'C:/Users/Pablo/Google Drive/TFM/Images/'
    ref_images = []
    ref_labels = []
    images_flat = images.reshape(num_samples, height*width)
    for i in range(len(os.listdir(path))):
        imgs_flat = images_flat[i*61:(i+1)*61]
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(imgs_flat)
        for centroid in kmeans.cluster_centers_:
            ind = np.argmin(np.linalg.norm(imgs_flat - centroid, axis=1))
            ref_images.append(images[i*61+ind])
            ref_labels.append(i)
    return ref_images, ref_labels

def compute_ORB_features(imgs):
    keypts = []
    descs = []
    for img in imgs:
        k, d = orb.detectAndCompute(img, None)
        keypts.append(k)
        descs.append(d)
    return keypts, descs

def align_image(im):
    keypts, desc = compute_ORB_features([im])
    keypts = keypts[0]
    desc = desc[0]
    if desc is not None and len(desc) > MAX_MATCHES*0.85:
        ims_aligned = []
        for e in future_lands:
            for j, elem in enumerate(ref_descs[e*REFS_PER_LAND:(e+1)*REFS_PER_LAND]):
                matches = matcher.match(desc, elem, None) # match features
                matches.sort(key=lambda x: x.distance, reverse=False) # sort matches by score
                matches = matches[:int(len(matches)*GOOD_MATCH_PERCENT)] # remove not so good matches
                # Extract location of good matches
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)
                for i, match in enumerate(matches):
                    points1[i, :] = keypts[match.queryIdx].pt
                    points2[i, :] = ref_keypts[e*REFS_PER_LAND+j][match.trainIdx].pt
                # Find homography
                hom, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
                if hom is not None:
                    img = cv2.warpPerspective(im, hom, (w, h))
#                    img[img==0] = int(round(np.mean(im))) # fill black pixels with mean values
                    ims_aligned.append(img)
                else:
                    ims_aligned.append(im*0)
        return ims_aligned
    else:
        return None

def get_nearest_neighbor(im):
    im_orig = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgs_aligned = align_image(im_orig)
    if imgs_aligned is not None:
        min_dist = np.zeros(len(imgs_aligned))
        pred_class = []
        for ind, im in enumerate(imgs_aligned):
            im = cv2.resize(im, (90,160))
            im_projected = np.dot(eigen_imgs.T, (im.reshape(1,160*90)-mean_vector).T).reshape(1,num_components)
            dists = np.zeros(len(future_lands))
            for i, e in enumerate(future_lands):
                dists[i] = np.min(np.linalg.norm(projections[e*61:(e+1)*61] - im_projected, axis=1))
            min_dist[ind] = np.min(dists)
            pred_class.append(future_lands[np.argmin(dists)])
        j = np.argmin(min_dist)
        pred_class = pred_class[j]
        min_dist = np.min(min_dist)
#        cv2.imshow('a',cv2.resize(np.hstack((ref_images[pred_class*REFS_PER_LAND+j%REFS_PER_LAND], im_orig, imgs_aligned[j])), (90*2*3,160*2)))
#        cv2.imshow('a',cv2.resize(np.hstack((ref_images[pred_class*REFS_PER_LAND+j%REFS_PER_LAND], im_orig, imgs_aligned[j])), (90*3,160)))
#        cv2.waitKey(1)
        return pred_class, min_dist
    else:
        return None, None

def change_future_lands_1(curr_land):
    if frame > 250:
        curr_land = 'HallBack'
    if frame > 338:
        curr_land = 'HallFront'
    if frame > 715:
        curr_land = 'Corridor'
    if frame > 1085:
        curr_land = 'DoorClose'
    return curr_land

def change_future_lands_2(curr_land):
    if frame > 208:
        curr_land = 'HallBack'
    if frame > 286:
        curr_land = 'HallFront'
    if frame > 600:
        curr_land = 'Corridor'
    if frame > 975:
        curr_land = 'DoorOpen'
    if frame > 1119:
        curr_land = 'Classroom'
    return curr_land

def change_future_lands_3(curr_land):
    if frame > 195:
        curr_land = 'HallBack'
    if frame > 280:
        curr_land = 'HallFront'
    if frame > 610:
        curr_land = 'Corridor'
    if frame > 935:
        curr_land = 'DoorClose'
    return curr_land

def change_future_lands_4(curr_land):
    if frame > 135:
        curr_land = 'HallBack'
    if frame > 178:
        curr_land = 'HallFront'
    if frame > 500:
        curr_land = 'Corridor'
    if frame > 824:
        curr_land = 'DoorClose'
    return curr_land

def change_future_lands_5(curr_land):
    if frame > 157:
        curr_land = 'HallBack'
    if frame > 239:
        curr_land = 'HallFront'
    if frame > 579:
        curr_land = 'Corridor'
    if frame > 851:
        curr_land = 'DoorClose'
    return curr_land

def check_lands_eigen(last_land, c, d):
    if last_land == 'Elevator' and c == 5 and d < 4500:
        last_land = 'HallBack'
    elif (last_land == 'Elevator' or last_land == 'HallBack') and c == 6 and d < 3500:
        last_land = 'HallFront'
    elif (last_land == 'HallBack' or last_land == 'HallFront') and c == 1 and d < 2300:
        last_land = 'Corridor'
    elif (last_land == 'HallFront' or last_land == 'Corridor') and c == 2 and d < 3000:
        last_land = 'DoorClose'
    elif (last_land == 'HallFront' or last_land == 'Corridor') and c == 3 and d < 2000:
        last_land = 'DoorOpen'
    elif (last_land == 'Corridor' or last_land == 'DoorOpen') and c == 0 and d < 2000:
        last_land = 'Classroom'
    return last_land

def check_lands_fisher(last_land, c, d):
    if last_land == 'Elevator' and c == 5 and d < 5000:
        last_land = 'HallBack'
    elif (last_land == 'Elevator' or last_land == 'HallBack') and c == 6 and d < 4380:
        last_land = 'HallFront'
    elif (last_land == 'HallBack' or last_land == 'HallFront') and c == 1 and d < 3000:
        last_land = 'Corridor'
    elif (last_land == 'HallFront' or last_land == 'Corridor') and c == 2 and d < 5200:
        last_land = 'DoorClose'
    elif (last_land == 'HallFront' or last_land == 'Corridor') and c == 3 and d < 5000:
        last_land = 'DoorOpen'
    elif (last_land == 'Corridor' or last_land == 'DoorOpen') and c == 0 and d < 2700:
        last_land = 'Classroom'
    return last_land
    

# =============================================================================
# LOAD IMAGES
# =============================================================================
path = 'C:/Users/Pablo/Google Drive/TFM/'
print('Loading images from dataset')
images, labels = load_images_and_labels(path + 'Images/')
num_samples, height, width = images.shape


# =============================================================================
# LOAD REFERENCE IMAGES AND COMPUTE ORB DESCRIPTORS
# =============================================================================
print('Computing reference images for alignement: ' + str(REFS_PER_LAND) + ' per landmark')
if not os.path.isfile(path + 'Models/' + 'ref_images_' + str(REFS_PER_LAND) + '.dat'):
    ref_images, ref_labels = load_ref_images(REFS_PER_LAND)
    pickle.dump((ref_images, ref_labels), open(path + 'Models/' + 'ref_images_' + str(REFS_PER_LAND) + '.dat', 'wb'))
else:
    ref_images, ref_labels = pickle.load(open(path + 'Models/' + 'ref_images_' + str(REFS_PER_LAND) + '.dat', 'rb'))
orb = cv2.ORB_create(MAX_MATCHES)
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
print('Computing ORB descriptors for reference images')
ref_keypts, ref_descs = compute_ORB_features(ref_images)

# =============================================================================
# LOAD MODIFIED IMAGES
# =============================================================================
print('Loading reduced images')
images, labels = load_images_and_labels(path + 'Modified Images/')

# =============================================================================
# SET PARAMETERS
# =============================================================================
if model_name == 'eigen':
    num_components = 40
    check_lands = check_lands_eigen
elif model_name == 'fisher':
    num_components = 6
    check_lands = check_lands_fisher


# =============================================================================
# CREATE/LOAD RECOGNIZER
# =============================================================================
print('Loading ' + model_name.upper()[0] + model_name[1:] + 'landmarks recognizer')
if not os.path.isfile(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml'):
    if model_name =='eigen':
        model = cv2.face.EigenFaceRecognizer_create(num_components)
    elif model_name == 'fisher':
        model = cv2.face.FisherFaceRecognizer_create()
    model.train(images, labels)
    model.write(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml')
else:
    if model_name =='eigen':
        model = cv2.face.EigenFaceRecognizer_create(num_components)
    elif model_name == 'fisher':
        model = cv2.face.FisherFaceRecognizer_create()
    model.read(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml')

# Get eigenvectors, mean and projections
eigen_imgs = model.getEigenVectors()
mean_vector = model.getMean()
projections = np.array(model.getProjections()).reshape(num_samples, num_components)
classes = model.getLabels().ravel()


# =============================================================================
# START NAVIGATOR
# =============================================================================
path = 'C:/Users/Pablo/Google Drive/TFM/Videos/'
change_future_lands_arr = [change_future_lands_1, change_future_lands_2, change_future_lands_3, change_future_lands_4, change_future_lands_5]
print('Starting navegator')

for video_name in video_names:
    # Open json file to store data
    f = open(path + 'labels_' + model_name + '_' + video_name[:-4] + '_' + technique + '_' + str(num_components) + '.json', 'w')
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
        w, h = h, w
    else:
        transpose = False
    
    change_future_lands = change_future_lands_arr[int(video_name[-5])-1]
    
    frame = 0
    
    # Store first frame's prediction
    st, im = cap.read()
    if im is not None:
        if transpose:
            im = rotate_bound(im, 90)
        if future_lands:
            cl, dis = get_nearest_neighbor(im)
        
        frame += 1
            
        # Store data into json file
        info = [frame, cl, dis]
        f.write(str(dict(zip(keys, info))).replace("'", '"'))
        frame += 1
        cl_prev = cl
        dis_prev = dis
        
    while(True):
        # Read next frame
        st, im = cap.read()
        if im is None:
            break
        if transpose:
            im = rotate_bound(im, 90)
        if future_lands:
            cl, dis = get_nearest_neighbor(im)
            if cl is None:
                cl, dis = cl_prev, dis_prev
        # Store data into json file
        info = [frame, cl, dis]
        f.write(',\n' + str(dict(zip(keys, info))).replace("'", '"'))
        frame += 1
#        current_land = change_future_lands(current_land)
        current_land = check_lands(current_land, cl, dis)
        future_lands = landmarks[current_land][1]
        cl_prev = cl
        dis_prev = dis
        
        if frame % 100 == 0:
            print(frame)
    
    f.write('\n]')
    f.close()
    cap.release()

# =============================================================================
# COMPARE DATA WITH REAL VALUES
# =============================================================================
accuracies_eucl = np.zeros((len(video_names)), dtype=np.float)

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
    data = np.array(list(map(lambda x: [x['frame'], x['c1'], x['d1']], data)))
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
    accuracies_eucl[i] = 100*float(predicted_frames_eucl)/total_real_frames
    
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
    data = np.array(list(map(lambda x: [x['frame'], x['c1'], x['d1']], data)))
    order = np.argsort(data, axis=0)[:,0] # order from initial frame to last frame
    data = data[order]
    max_dist1 = data[:,2].max()
    
    # Plot predicted labels and distances to nearest neighbors
    ax1 = plt.subplot('11'+str(i+1))
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





