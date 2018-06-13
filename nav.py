# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:28:22 2018

@author: Pablo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:22:20 2018

@author: Pablo
"""
import os
from sklearn.cluster import KMeans
from imutils import rotate_bound
import cv2
import numpy as np
from re import sub as re_sub
import pickle
from pdb import set_trace as STOP
import time

video_name = 'Video4.mp4'

model_name = 'eigen'
num_components = 40
technique = 'nav'
REFS_PER_LAND = 5
MAX_MATCHES = 300
GOOD_MATCH_PERCENT = 0.05
path_mvt = 'C:/Users/Pablo/Google Drive/TFM/MVT/'
landmarks = {"Classroom": [0,None,],
             "Corridor": [1,[2,3,0]],
             "DoorClose": [2,None],
             "DoorOpen": [3,[0]],
             "Elevator": [4,[5,6]],
             "HallBack": [5,[6,1]],
             "HallFront": [6,[1,2,3]]}
for elem_name in landmarks.keys():
    im = cv2.resize(cv2.imread(path_mvt + elem_name + '.png'),(353,156))
    landmarks[elem_name].append(im) # mvt image
    landmarks[elem_name].append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) > -1) # mvt image mask
    
current_land = "Elevator"
future_lands = landmarks[current_land][1]
mvt = landmarks[current_land][2]
mask = landmarks[current_land][3]


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
        return pred_class, min_dist
    else:
        return None, None

def check_lands(last_land, c, d):
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
# CREATE/LOAD RECOGNIZER
# =============================================================================
print('Loading ' + model_name.upper()[0] + model_name[1:] + 'landmarks recognizer')
model = cv2.face.EigenFaceRecognizer_create(num_components)
if not os.path.isfile(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml'):
    model.train(images, labels)
    model.write(path + 'Models/' + model_name + '_' + technique + '_' + str(num_components) + '.yml')
else:
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
print('Starting navigator')

# Load video
cap = cv2.VideoCapture(path + video_name)
if not cap.isOpened():
    print('Error opening video file')

w, h = (int(cap.get(3)), int(cap.get(4))) # w, h = (720, 1280)
if w > h:
    transpose = True
    w, h = h, w
else:
    transpose = False

frame = 0
cl_prev = 4
dis_prev = 0
prev_land = 'Elevator'
times = []

while(True):
    st, im = cap.read()
    if im is None:
        break
    if transpose:
        im = rotate_bound(im, 90)
    tic = time.time()
    if future_lands:
        cl, dis = get_nearest_neighbor(im)
        if cl is None:
            cl, dis = cl_prev, dis_prev
    frame += 1
    
    current_land = check_lands(current_land, cl, dis)
    if current_land != prev_land:
        future_lands = landmarks[current_land][1]
        mvt = landmarks[current_land][2]
        mask = landmarks[current_land][3]
        prev_land = current_land
    cl_prev = cl
    dis_prev = dis
    times.append(time.time() - tic)
    
#    im[:156,-353:,:] = mvt
#    cv2.imshow('a',im)
#    cv2.waitKey(1)

cap.release()
avg_time = sum(times) / len(times)
print('Average frame processing time: ' + str(avg_time))