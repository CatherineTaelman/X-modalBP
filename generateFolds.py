"""
This file contains the following functions:
    1) loadData(dataset, useFeatures)
       Loads dataset (including extra features if useFeatures=True).
       Returns X_total:  labelled image pixels, including their spatial coordinates
                         (last two entries in spectral dimension are the spatial coordinates)
               y_total:  ground truth (labels) corresponding to the pixels in X_total
               dimHS:    spectral dimension of the hyperspectral data
               dimLidar: spectral dimension of the Lidar data
               TO DO: change dimHS and dimLidar to dim1 and dim2 (spectral dim of modality 1 & modality 2)
               
    2) generateKFolds(x, y, nr_folds, useFeatures):
       Generates k folds in a stratified fashion.
       Returns train: list containing the k different train subsets
               test:  list containing the k different test subsets
               
    3) generateFoldDictionary(dataset, nr_folds, useFeatures):
       Calls the functions loadData and generateKFolds and collects the output in a dictionary 'folds'.
       Returns folds: a dictionary with the following keys: 'X_total', 'train', 'test', 'nr_folds', 'dimHS', 'dimLidar'. 

       This method is created to deal with the scalability issue encountered during label propagation.
       Too large data sets must be split into k folds, which are then processed one-by-one by the label prop. method.
       Required amount of folds (nr_folds) depends on size of data set. For Trento and Houston data sets nr_folds=6 folds is used.

Author: Catherine Taelman
June 2021
For questions or suggestions: taelman.catherine@gmail.com
"""
import numpy as np
import tifffile
from skimage import img_as_ubyte, img_as_float
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import sys
import scipy.io
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric
from scipy.sparse import csr_matrix, eye
from sklearn.metrics import pairwise
import itertools
import pickle
from read_write_img import get_all_bands

from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
realDataDir = '/work/cta014/data'

def loadData(dataset, useFeatures=False):
# returns labelled pixels including their spatial coordinates (X_total) with corresponding ground truth (y_total)
# also returns spectral dimension of the hyperspectral and lidar data (dimHS & dimLidar)
# if useFeatures=True, then method will add LiDAR-extracted features (from GLCM) to original LiDAR image and use
# only selected features from HS data (selected via GKMI algorithm). Default is False.
    def normalize(array):
            # Normalize the data with range from 0 to 1
            array = img_as_float(array)
            return np.array((array - np.min(array)) / (np.max(array) - np.min(array)))
        
    if dataset == 'Trento':
        def importData(filename):
                input_images = tifffile.imread(join(realDataDir, dataset, filename))
                dim1 = len(input_images) # depth dimension (z)
                dim2 = len(input_images[0]) # spatial dim (x)
                dim3 = len(input_images[0][0]) # spatial dim (y)
                input_images = np.reshape(input_images,(dim1,dim2*dim3))
                input_images = np.transpose(input_images)
                input_images = np.reshape(input_images,(dim2,dim3,dim1)) # (166,600,63)
                # def normalize(array):
                #     # Normalize the data with range from 0 to 1
                #     array = img_as_float(array)
                #     return np.array((array - np.min(array)) / (np.max(array) - np.min(array)))
                input_images = normalize(input_images)
                return input_images
    
        # load ground truth
        mat1 = scipy.io.loadmat(join(realDataDir, dataset, 'Trento_TrainSamples.mat')) 
        mat2 = scipy.io.loadmat(join(realDataDir, dataset, 'Trento_TrainAndTestSamples.mat')) 
        gt_train = mat1['TNsecSUBS_Train'] 
        gt_total = mat2['TNsecSUBS_Test'] 
        gt_test = gt_total - gt_train
    
        # load original HS and LiDAR images
        if useFeatures == False:
            input_images_hyper = importData(filename='Trento_hyperspectral.tif')
            input_images_lidar = importData(filename= 'Trento_LiDAR.tif')
            input_images_lidar = np.reshape(input_images_lidar[:,:,0],(input_images_lidar.shape[0],input_images_lidar.shape[1],1))
    
        # load selected and extracted features from HS and LiDAR data
        else:
            input_images_hyper = np.load('Trento_hyperspectral_GKMI_Features.npy') # using features selected via GKMI
            input_images_hyper = normalize(input_images_hyper)
            input_images_lidar = importData(filename= 'Trento_LiDAR.tif')
            input_images_lidar = np.reshape(input_images_lidar[:,:,0],(input_images_lidar.shape[0],input_images_lidar.shape[1],1))
            lidarFeat = importData(filename = 'Trento_LiDAR_GLCM_Features_band1.tif') # LiDAR texture features extracted from GLCM
            input_images_lidar = np.concatenate((input_images_lidar,lidarFeat), axis = -1 )
    

        dimHS = input_images_hyper.shape[2] # spectral dimension of HS image
        dimLidar = input_images_lidar.shape[2]
        print('hyperspectral input shape: ', input_images_hyper.shape)
        print('LiDAR input shape: ', input_images_lidar.shape)
        
        # stack HS and Lidar data into one feature vector
        X_total = np.concatenate((input_images_hyper, input_images_lidar), axis=-1) 
        row_combi, col_combi, f_num_combi = X_total.shape
        
        # take only labelled pixels and split data into train-test set
        idx_known = gt_test != 0
        X_total = X_total[idx_known] # this is new data set (only labelled points taken into account)
        y_total = gt_test[idx_known] # this is new ground thruth
        print('shape of ground truth: ', y_total.shape)
        X_total_coords = np.argwhere(idx_known == True) # x,y coords of known pixels
        X_total = np.concatenate((X_total, X_total_coords),axis=1)
    
    elif dataset == 'Houston':
        X_total = scipy.io.loadmat(join(realDataDir, dataset,'2013_IEEE_GRSS_DF_Contest_InputImages.mat'))['features']
        row_combi, col_combi, f_num_combi = X_total.shape
        y = tifffile.imread(join(realDataDir,dataset, '2013_IEEE_GRSS_DF_Contest_TestSamples.tif')) # test ground truth image
        y = np.transpose(y)
        print('shape of ground truth test: ', y.shape)
        
        if useFeatures==False:
            input_images_lidar = X_total[:,:,6]
            input_images_lidar = normalize(input_images_lidar)
            input_images_lidar = np.reshape(input_images_lidar,(input_images_lidar.shape[0],input_images_lidar.shape[1],1))
            input_images_hyper = X_total[:,:,7:]
            input_images_hyper = normalize(input_images_hyper)
            
        else:
            # import original LiDAR image
            input_images_lidar = X_total[:,:,6]
            input_images_lidar = normalize(input_images_lidar)
            input_images_lidar = np.reshape(input_images_lidar,(input_images_lidar.shape[0],input_images_lidar.shape[1],1))
            
            # import texture LiDAR features (extracted from GLCM)
            lidarFeatures = tifffile.imread(join(realDataDir, dataset, '2013_IEEE_GRSS_DF_Contest_LiDAR_GLCM_Features.tif'))
            lidarFeatures = np.transpose(lidarFeatures, (2, 1, 0))
            # replace NaN values in last image with 0's (football field is NaN)
            lidarFeatures[np.isnan(lidarFeatures)]=0
            lidarFeatures = normalize(lidarFeatures)
            input_images_lidar = np.concatenate((input_images_lidar,lidarFeatures), axis = -1 )
            # import HS features selected via GKMI
            input_images_hyper = np.load(join(realDataDir, dataset, '2013_IEEE_GRSS_DF_Contest_Hyper_GKMI_Features.npy')) # use 50 features selected via GKMI
            input_images_hyper = normalize(input_images_hyper)
            
        dimLidar = input_images_lidar.shape[2]
        dimHS = input_images_hyper.shape[2] # spectral dimension of HS image
        print('lidar input shape: ', input_images_lidar.shape)
        print('hyperspectral input shape: ', input_images_hyper.shape)
        
        X_total = np.concatenate((input_images_hyper, input_images_lidar), axis=-1) # stack HS and LiDAR data
        row_combi, col_combi, f_num_combi = X_total.shape
        
        # keep only labelled pixels and add spatial coordinates of each pixel --> store this in X_total
        idx_known = y != 0
        X_total = X_total[idx_known] # this is new data set (only labelled points taken into account)
        y_total = y[idx_known] # this is new ground thruth
        X_total_coords = np.argwhere(idx_known == True) # x,y coords of known pixels
        X_total = np.concatenate((X_total, X_total_coords),axis=1)

    elif dataset == 'seaIce':
        X_total = tifffile.imread(join(realDataDir,'scaled_images/S1A_EW_GRDM_1SDH_20150327T115532_20150327T115632_005217_00696B_D22F_scaled_cropped.tif'))
        #print('shape of original X_total:', X_total.shape)
        
        y = get_all_bands(join(realDataDir,'training_masks/S1A_EW_GRDM_1SDH_20150327T115532_20150327T115632_005217_00696B_D22F_training_mask.img'))
        y = y.astype(int)
        #print('shape of original y:', y.shape)
        
        dimHS=1 # no HS and Lidar data here, but 2 different bands, each with spectral dimension 1
        dimLidar=1
        
        # keep only labelled pixels and add spatial coordinates of each pixel --> store this in X_total
        idx_known = y != 0
        X_total = X_total[idx_known] # this is new data set (only labelled points taken into account)
        y_total = y[idx_known] # this is new ground thruth
        
        print('shape of ground truth (y) only known pixels:', y_total.shape)
        X_total_coords = np.argwhere(idx_known == True) # x,y coords of known pixels
        X_total = np.concatenate((X_total, X_total_coords),axis=1)
        
    else:
        print('non-existing dataset specified, check user input')
    
    return X_total, y_total, dimHS, dimLidar
 
     
def generateKFolds(X, y, nr_folds, useFeatures = False):
# returns train and test list containing train and test subsets for each fold
    skf = StratifiedKFold(n_splits=nr_folds, shuffle=True)
    train = [] # collect training samples (X and y) of all folds in list
    test = [] # collect test samples (X and y) of all folds in a list
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train.append(X_train)
        train.append(y_train)
        test.append(X_test)
        test.append(y_test)
    return train, test

def generateFoldDictionary(dataset, nr_folds, useFeatures=False):
# returns a dictionary called 'folds' containing: labelled pixels including their spatial coordinates (X_total),
# lists of train and test folds, amount of folds (nr_folds), dimHS and dimLidar
    X_total, y_total , dimHS, dimLidar = loadData(dataset, useFeatures)
    train, test = generateKFolds(X_total, y_total, nr_folds)
    folds = {'X_total': X_total, 'train':train, 'test':test, 'nr_folds': nr_folds, 'dimHS': dimHS, 'dimLidar': dimLidar}
    return folds

#X_total, y_total, dimHS, dimLidar = loadData(dataset='seaIce')

#folds = generateFoldDictionary(dataset = 'Houston', nr_folds = 6)
# dataset='Houston'
# pickle.dump(folds, open("folds_"+dataset+".pickle", "wb"))