from Dataloader_utils import *
import numpy as np
from sklearn import svm
from test import print_reporting
import argparse
import cv2

if __name__ == '__main__':
    '''
    [feature_option] 
    if you want SIFT feature, plz press 'SIFT'
    if you want HOG feature, plz press 'HOG'

    [kernel_option]
    if you want Linear kernel, plz press 'linear'
    if you want Gaussian kernel, plz press 'rbf'
    
    [Other hyperparameter]
    --c is parameter which is used in Support Vector Machine , default value is 1.
    --gamma is parameter which is used in Support Vector Machine , default value is 0.008.
    --b is background_delete parameter, default is True. 
    '''
    # make instance
    parser = argparse.ArgumentParser(description='SIFT and HOG testing')

    # argument setting
    parser.add_argument('--feature', type=str, help='SIFT = SIFT, HOG = HOG')
    parser.add_argument('--kernel', type=str, help='Linear = linear, Gaussian = rbf')
    parser.add_argument('--c', type=int, required=False, default=1, help='Linear = linear, Gaussian = rbf')
    parser.add_argument('--gamma', type=float, required=False, default=0.008, help='Linear = linear, Gaussian = rbf')
    parser.add_argument('--b', type=bool, required=False, default=True, help='Linear = linear, Gaussian = rbf')

    # save arguments
    args = parser.parse_args()

    # Dataload
    print('[Dataset loading...]')
    train_dataset_path = './plant-seedlings-classification/train/'
    train_dataset, valid_dataset = Dataset(train_dataset_path, background_delete = args.b)
    X_train, y_train, X_test, y_test = Dataloder(train_dataset, valid_dataset, args.feature)


    #training
    print(f'Model option: [{args.kernel}]')
    clf = svm.SVC(kernel=args.kernel, C = 1, random_state=42, gamma = 0.008)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    print('---[reporting]---')
    print_reporting(y_test, prediction)









