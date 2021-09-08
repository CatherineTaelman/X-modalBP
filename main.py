"""
MULTIMODAL LABEL PROPAGATION ALGORITHM FOR REMOTE SENSING DATA
More info in README file.

User should input:
    1)  dataset to use ('Trento' or 'Houston')
    2) 'choice': which case to run (case 1: 'full overlap', case 2: 'hyperspectral' or 'lidar')

Author: Catherine Taelman
June 2021
For questions or suggestions: cta014@uit.no
"""

# -------------------------------------------------------------
""" ----  USER INPUT -----"""

dataset = 'Trento' # choose from ['Trento', 'Houston']
choice = 'full overlap' # choose from ['full overlap', 'hyperspectral', 'lidar']
realDataDir = '/work/cta014/data' #specify path to directory containing the data sets
# -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise
import pickle
from LCE import estimateH
from LCE_singleH import estimateSingleH
from numpy import linalg as LA
from sklearn.metrics import cohen_kappa_score
from generateFolds import generateFoldDictionary
from label_propagation import label_propagation


# define global variables k (amount of classes in ground truth) and nr_folds (amount of folds to split dataset into)
if dataset == 'Houston':
    k = 15
    nr_folds = 6
elif dataset == 'Trento':
    k = 6
    nr_folds = 6

## split data set into folds 
"""      PARAMETERS for generateFoldDictionary:
    
dataset:     use-defined input
nr_folds:    in how many folds the data set needs to be split up
useFeatures: if True, then method will add LiDAR-extracted features (from GLCM) to original LiDAR image and use
             only selected features from HS data (selected via GKMI algorithm).
             Default is False.
"""
folds = generateFoldDictionary(dataset, nr_folds, realDataDir, useFeatures=False)
testFolds = folds['test']
nr_folds = folds['nr_folds']
dimHS = folds['dimHS'] # spectral dimension of hyperspectral image
dimLidar = folds['dimLidar'] # spectral dimension of LiDAR image
X_total = folds['X_total'] # complete input data (different modalities stacked together in one feature vector)
print('dim1 (dimHS) = ',dimHS)
print('dim2 (dimLiDAR) = ', dimLidar)
print('-- dataset split into '+str(nr_folds)+' folds --')

# load optimal hyperparameters 
rs = pickle.load(open('optimal_rs_'+dataset+'.pickle', "rb")) 
print('-- loaded optimal hyperparameters --')

j = 1
acc_list = []
kappa_list = []
conf_matrices = []

if choice == 'full overlap':
    print('-- running LP on', dataset+' in case of '+choice+" --")
elif choice == 'hyperspectral' or 'lidar':
    print('-- running LP on', dataset+' in case of partial overlap + '+choice+" only --")
    
# loop over folds of test set (running over all test folds = running over every pixel of global image once)
for i in range(0, len(testFolds), 2):
    print(' Test fold ', j)
    X = testFolds[i]
    y = testFolds[i+1]
    coordinates = X[:,-2:]
    finals_list = []
    # run label propagation x times (5 recommended) for each fold, using optimal hyperparameters as input
    for l in range(2):
        acc, kappa, conf, finals = label_propagation(X, y, X_total, k, dimHS, dimLidar, kNN = rs['kNN'], ep = rs['ep'], sig_spe = rs['sig_spe'], sig_spa = rs['sig_spa'],
                                 r = rs['r'], choice = choice)
        acc_list.append(acc)
        kappa_list.append(kappa)
        conf_matrices.append(conf)
        finals_list.append(finals)
     
    # for each fold save final beliefs and corresponding coordinates (for classification map)
    results_labels = {'final beliefs': finals_list, 'coordinates': coordinates, 'acc_list': acc_list, 'kappa_list':kappa_list,
                'conf_matrices': conf_matrices}
    pickle.dump(results_labels, open("LP_Results_"+dataset+"_"+choice+"_Fold"+str(j)+".pickle", "wb"))
    j +=1

# save total results    
results = {'acc_list': acc_list, 'kappa_list':kappa_list,
                'conf_matrices': conf_matrices, 'final beliefs': finals_list}

print('saving results to ', "LP_Results_"+dataset+"_"+choice+".pickle")
pickle.dump(results, open("LP_Results_"+dataset+"_"+choice+".pickle", "wb"))
    

