import cv2
import os
import numpy as np
from  Preprocessing_utils import *
from skimage.feature import hog

def SIFT(img):
    img = cv2.resize(img, dsize=(128, 256), interpolation=cv2.INTER_CUBIC)
    sift = cv2.xfeatures2d.SIFT_create()
    _, des = sift.detectAndCompute(img, None)

    return des

def HOG(img):
    img = cv2.resize(img, dsize=(128, 256), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=False)
    return fd

def Dataset(train_dataset_path, background_delete=False):
    listdir = os.listdir(train_dataset_path)
    listdir = [listdir for listdir in listdir if not listdir.startswith('.')]  # .DS_Store delete

    train_dataset = []
    valid_dataset = []

    for i, cl_name in enumerate(listdir):
        imgdir = os.listdir(train_dataset_path + cl_name)
        train_num = 0
        for num, img_name in enumerate(imgdir):
            img = cv2.imread(train_dataset_path + cl_name + '/' + img_name, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if background_delete == True:
                img =  delete_Background(img)

            tmp = [cl_name, img]
            if num < int(len(imgdir) * 0.8):
                train_dataset.append(tmp)
                train_num += 1
            else :
                valid_dataset.append(tmp)
        print('['+str(i+1) +'. '+ cl_name +', [total_data]: ' + str(len(imgdir)) + ' / [train_data]: ' +str(train_num) + ']')
    return train_dataset, valid_dataset



def Dataloder(train_dataset, valid_dataset, option):
    t_labels = []
    v_labels = []
    tmp_list_vec = []
    t_fd_list=[]
    v_fd_list = []
    tmp_list=[]
    if option == 'SIFT':
        #train_dataloder
        for i, (cl_name, img) in enumerate(train_dataset):
            fd = SIFT(img)
            if fd is not None:
                tmp_list.append(fd)
                t_labels.append(cl_name)
        # make it one line
        for descriptor in tmp_list:
            if descriptor is not None:
                for des in descriptor:
                    tmp_list_vec.append(des)

        num_cluster = 60
        BoW = kmean_bow(tmp_list_vec, num_cluster)
        t_fd_list = create_feature_bow(tmp_list, BoW, num_cluster)

        tmp_list=[]
        # valid_dataloder
        for i, (cl_name, img) in enumerate(valid_dataset):
            fd = SIFT(img)
            if fd is not None:
                tmp_list.append(fd)
                v_labels.append(cl_name)
        #use train_dataset BOW
        v_fd_list = create_feature_bow(tmp_list, BoW, num_cluster)


    elif option == 'HOG':
        for i, (cl_name, img) in enumerate(train_dataset):  # 3794 times rotate
            fd = HOG(img)
            if fd is not None:
                t_fd_list.append(fd)
                t_labels.append(cl_name)

        for i, (cl_name, img) in enumerate(valid_dataset):  # 3794 times rotate
            fd = HOG(img)
            if fd is not None:
                v_fd_list.append(fd)
                v_labels.append(cl_name)

    return t_fd_list, t_labels, v_fd_list, v_labels
