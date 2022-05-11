import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# this mask filter function is cited and transformed by "https://www.kaggle.com/code/wong1131/limmy-s"
# each steps is described below
'''
[Delete Background by using using masking method with HSV feature]
1st picture: original
2nd image: mask, based on HSV filter to select green
3rd image: original image with mask applied
4th image: application of a Gaussian filter on the 3rd image
'''

def create_mask_for_plant(image):
    sensitivity=35
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


def delete_Background(image):
    image_segmented = segment_plant(image)
    image_sharpen = sharpen_image(image_segmented)

    return image_sharpen

def show_image_preprocc(image):
    image_mask = create_mask_for_plant(image)
    image_segmented = segment_plant(image)
    image_sharpen = sharpen_image(image_segmented)
    fig, axs = plt.subplots(1, 4, figsize=(20, 20))
    axs[0].imshow(image)
    axs[0].set_title('origin_img')
    axs[1].imshow(image_mask)
    axs[1].set_title('masked_img')
    axs[2].imshow(image_segmented)
    axs[2].set_title('segmented_img')
    axs[3].imshow(image_sharpen)
    axs[3].set_title('GaussianBlur_img')


def flatten_des(des, size_clipper = None):

    des = des.astype(np.float64)
    np.random.shuffle(des)

    if des.shape[0]>size_clipper:
        if size_clipper is not None :
            des = des[0:size_clipper, :]  # trim vector so all are same size

    vector_data = des.reshape(1, des.shape[0] * des.shape[1])

    return vector_data

def kmean_bow(all_descriptors, num_cluster):

    kmeans = KMeans(n_clusters = num_cluster, random_state=42)
    kmeans.fit(all_descriptors)

    bow_dict = kmeans.cluster_centers_

    return bow_dict

def create_feature_bow(image_descriptors, BoW, num_cluster):

    X_features = []

    for i in range(len(image_descriptors)):
        features = np.array([0] * num_cluster)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)

            argmin = np.argmin(distance, axis = 1)

            for j in argmin:
                features[j] += 1
        X_features.append(features)

    return X_features
