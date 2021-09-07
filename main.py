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

from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))


"""     PARAMETERS for main_LP:

X:           the fold containting data points to label including their spatial coordinates (last two entries in spectral dimension)
y:           ground truth (corresponding to the data points in the fold X)
X_total:     feature vector where all modalities are stacked together for full data set (not split into folds)
k:           amount of classes in the ground truth (k=6 for Trento, k=15 for Houston)
kNN:         k-nearest neightbours for calculating W3 (affinity matrix between nodes of type 1 and type 2)
ep:          interaction strength, inherent to original ZooBP label prop. method (range is bounded to guarantee convergence)
sig_spe:     width of the Gaussian kernel in spectral dimension (used to calculate affinity matrices W1 and W2 according to Shi&Malik formula)
sig_spa:     width of the kernel in spatial dimension (used to calculate affinity matrices W1 and W2 according to Shi&Malik formula)
r:           amount of pixels away to consider when defining affinity matrices W1 and W2
choice:      ['full overlap', 'lidar', 'hyperspectral'] --> choose either fully overlapping case, or partial overlap (either lidar or hyperspectral only for 1/4th of the image)
fraction:    fraction of the labels in y that are set to label 0 to mimic unlabelled samples (note: seed size = 1 - percentage)
             Default is set to 0.8 (meaning 20% of nodes are randomly set to their ground thruth label at start)
             
        PARAMETERS for generateFoldDictionary:
    
dataset:     use-defined input
nr_folds:    in how many folds the data set needs to be split up
useFeatures: if True, then method will add LiDAR-extracted features (from GLCM) to original LiDAR image and use
             only selected features from HS data (selected via GKMI algorithm).
             Default is False.
"""

# define global variables k (amount of classes in ground truth) and nr_folds (amount of folds to split dataset into)
if dataset == 'Houston':
    k = 15
    nr_folds = 6
elif dataset == 'Trento':
    k = 6
    nr_folds = 6

def main_LP(X, y, X_total, k, dimHS, dimLidar, kNN, ep, sig_spe, sig_spa, r, choice, fraction=0.8):
    assert len(X) == len(y), 'X and y must have same length'
    
    if choice != 'full overlap':
        split = round(len(X)* 3/4) # take 3/4rd of data for overlapping part, one quarter for non-overlapping
        X_overlap = X[0:split,:-2] # take first 3/4rd of (labelled) pixels for combi set
        X_overlap_coords = X[0:split,-2:]
        
    else:
        X_overlap = X[:,:-2]
        X_overlap_coords = X[:,-2:]
    
    if choice == 'lidar':
        X_lidar = np.reshape(X[:,dimHS:dimHS+dimLidar],(len(X),1)) # all LiDAR data
        #print('shape of lidar array in X:', X_lidar.shape)
        # make lidar data equivalent in size to combi data (pad with zeros to have same depth dimension)
        def padarray(A, size):
            t = size - len(A)
            return np.pad(A, pad_width=(0, t), mode='constant')
        
        nonOverlap_lidar = X_lidar[split:] #take last quarter of pixels from LiDAR image
        nonOverlap_lidar_coords = X[split:, -2:]
        nonOverlap_padded = np.zeros((len(nonOverlap_lidar), X_total.shape[1] - 2))
        i = 0
        for row in nonOverlap_lidar:
            padded_row = padarray(row, X_total.shape[1] - 2)
            nonOverlap_padded[i,:] = padded_row 
            i = i + 1
        
    if choice == 'hyperspectral':
        X_hyper = X[:,0:dimHS] # all HS data   604,63
        #print('shape of hyperspectral array in X: ', X_hyper.shape)
        
        # make HS data equivalent in size to combi data (pad with zeros to have same depth dimension)
        def padarray(A, size):
            t = size - len(A)
            return np.pad(A, pad_width=(0, t), mode='constant')
        
        nonOverlap_hyper = X_hyper[split:,:] # take last quarter of (labelled) pixels from hyperspectral image
        nonOverlap_hyper_coords = X[split:,-2:]   
        nonOverlap_padded = np.zeros((len(nonOverlap_hyper), X_total.shape[1] - 2))
        i = 0
        for row in nonOverlap_hyper:
            padded_row = padarray(row, X_total.shape[1] - 2)
            nonOverlap_padded[i,:] = padded_row 
            i = i + 1

        
    # set fraction of pixels to label 0 (meaning unlabelled), default fraction = 0.8
    gt_adjusted = np.copy(y)
    indices = np.random.choice(np.arange(gt_adjusted.size), replace=False,
                               size=int(gt_adjusted.size * fraction))
    gt_adjusted[indices] = 0
    
    if choice != 'full overlap':
        gt_adjusted_combi = gt_adjusted[0:split] #ground truth for combi data (overlapping region)
        gt_adjusted_nonOverlap = gt_adjusted[split:] # ground truth for lidar data OR for HS data in non-overlapping region
    
        # construct weigted adjacency matrices W1, W2 and W3 
        # W1 and W2 are constructed using spectral-spatial similarity metric of Shi&Malik
        S = len(X_overlap)
        W1 = np.zeros((S,S))
        for i in range (0,S):
            for j in range (i,S):
                normspe = LA.norm(X_overlap[i] - X_overlap[j])
                normspa = (LA.norm(X_overlap_coords[i] - X_overlap_coords[j]))
                
                if normspa < r:
                    W1[i,j] = np.exp(-(normspe**2/(sig_spe**2)))*np.exp(-(normspa**2/(sig_spa**2)))
                    W1[j,i] = W1[i,j]
                               
        S = len(X)-split
        W2 = np.zeros((S,S))
        for i in range (0,S):
            for j in range (i,S):
                if choice == 'hyperspectral':
                    normspe = LA.norm(X_hyper[i] - X_hyper[j])
                    normspa = (LA.norm(nonOverlap_hyper_coords[i] - nonOverlap_hyper_coords[j]))
                elif choice == 'lidar':
                    normspe = LA.norm(X_lidar[i] - X_lidar[j])
                    normspa = (LA.norm(nonOverlap_lidar_coords[i] - nonOverlap_lidar_coords[j]))
                if normspa < r:
                    W2[i,j] = np.exp(-(normspe**2/(sig_spe**2)))*np.exp(-(normspa**2/(sig_spa**2)))
                    W2[j,i] = W2[i,j]
                    
        # W3 is constructed using Gaussian kernel (spectral similarity) and kNN
        metric = pairwise.rbf_kernel(X_overlap, nonOverlap_padded, gamma=sig_spe) # Gaussian kernel
        sim = np.argsort(-metric, axis=1)
        mask = np.zeros(metric.shape)
        ind = sim[:,:kNN] # indices of k-nearest neighbours
        for mask_row, ind_row in zip(mask, ind):
            mask_row[ind_row] = 1
        W3 = metric.copy()
        W3[mask==0] = 0
        
        gtTypes= [gt_adjusted_combi, gt_adjusted_nonOverlap]
        classes = [k, k]
        
        N1 = gt_adjusted_combi.shape[0] # amount of nodes of type 1
        N2 = gt_adjusted_nonOverlap.shape[0] # amount of nodes of type 2
        nodes = [N1,N2]
        
    else:
    # in case of full overlap there is just a single affinity matrix W (computed with spectral-spatial similarity)
        S = len(X_overlap)
        W = np.zeros((S,S))
        for i in range (0,S):
            for j in range (i,S):
                normspe = LA.norm(X_overlap[i] - X_overlap[j])
                normspa = (LA.norm(X_overlap_coords[i] - X_overlap_coords[j]))
                
                if normspa < r:
                    W[i,j] = np.exp(-(normspe**2/(sig_spe**2)))*np.exp(-(normspa**2/(sig_spa**2)))
                    W[j,i] = W[i,j]
                    
        gtTypes= [gt_adjusted] 
        classes = [k , k]  
        
        N = gt_adjusted.shape[0] # amount of nodes
        nodes = [N]
                    
    # construct prior belief vector e: 3 steps
    E_total=[]
    for t, c in zip(gtTypes, classes):
        # step 1: reshape priors per node type into format (n_s x k_s) with 1 indicating it belongs to that class
        E = np.zeros([len(t), c])
        for i in range(len(t)):
            if t[i] != 0:
                pos = t[i] - 1
                E[i,pos] = c*0.001
                
        # step 2: for labeled nodes, initiate prior as k*0.001 for right class and -0.001 for wrong class       
        count = 0
        for roww in E:
            if np.max(roww)>0:
                roww = np.where(roww==0, -0.001, roww)
                E[count,:]=roww
            count = count + 1  
        E_total.append(E)
        
        # step 3: vectorize E  
    if choice != 'full overlap':
        e = np.concatenate((E_total[0].ravel(),E_total[1].ravel()))
    else:
        e = np.reshape(E, (E.shape[0]*E.shape[1],1)) 
    e = np.reshape(e, (e.shape[0],1)) 
    #print('check if e is residual (0-centered): average value is:', np.mean(e))
    
    # randomly initialize B or use: b = np.zeros((len(e),1)) 
    B_total=[]
    m = 1e-3
    for c, N in zip(classes, nodes):
        B = np.array([m*(np.random.uniform(size=(c,))-1/2) for j in range(N)]) #(k_s x n_s) matrix 
        B_total.append(B)
        
    if choice != 'full overlap':
        b = np.concatenate((B_total[0].ravel(),B_total[1].ravel()))
    else:
        b = np.reshape(B, (B.shape[0]*B.shape[1],1))
    b = np.reshape(b, (b.shape[0],1)) 
    #print('check if b is residual (0-centered): average value is:', np.mean(b))
    
    if choice != 'full overlap':
        E1 = E_total[0]
        E2 = E_total[1]
        Eupper = np.concatenate((E1,csr_matrix(np.shape(E1)).toarray()),axis=1)
        Elower = np.concatenate((csr_matrix(np.shape(E2)).toarray(),E2),axis=1)
        E_tot = np.concatenate((Eupper,Elower),axis=0)
        
        Wupper = np.concatenate((W1,W3), axis=1)
        Wlower = np.concatenate((np.transpose(W3),W2), axis=1)
        W_tot = np.concatenate((Wupper,Wlower),axis=0)
        
        # estimate potential matrices H via Linear Compatibility Estimation (LCE)
        # H1: combi-combi, H2: Lidar-lidar, H3: combi-lidar edges
        H_total = estimateH(E=E_tot, W=W_tot, k=k, method='LHE')
        H1 = np.transpose(H_total[0:k,0:k])
        H2 = np.transpose(H_total[k:,k:])
        H3 = H_total[0:k,k:]
        
        # construct P (for 2 node types and 3 edges types)
        P11 = (ep/k) * (np.kron(W1,H1))
        P12 = (ep/k) * (np.kron(W3,H3))
        P21 = (ep/k) * (np.kron(np.transpose(W3),np.transpose(H3)))
        P22 = (ep/k) * (np.kron(W2,H2))
        
        Pupper = np.concatenate((P11,P12),axis=1)
        Plower = np.concatenate((P21,P22), axis=1)
        P = np.concatenate((Pupper,Plower),axis=0) # persona-influence matrix P
        
        # define diagonal degree matrices 
        D11 = np.diag(np.sum(W1,1)) # degree matrix for nodes of type 1 connected with other nodes of type 1
        D22 = np.diag(np.sum(W2,1))
        D12 = np.diag(np.sum(W3,1)) # degree matrix for nodes of type 1 connected with nodes of type 2
        D21 = np.diag(np.sum(W3,0)) # degree matrix for nodes of type 2 connected with nodes of type 1
        
        # construct Q from D's and H's 
        Q1 = (ep/k)**2 * (np.kron(D11, np.matmul(H1,np.transpose(H1)))) + (ep**2/(k*k)) * (np.kron(D12, np.matmul(H3, np.transpose(H3))))
        Q2 = (ep/k)**2 * (np.kron(D22, np.matmul(H2,np.transpose(H2)))) + (ep**2/(k*k)) * (np.kron(D21, np.matmul(np.transpose(H3), H3)))
        # Q1 = eye(N1*k1) + (ep/k1)**2 * (np.kron(D11, np.matmul(H1,np.transpose(H1)))) + (ep**2/(k1*k2)) * (np.kron(D12, np.matmul(H3, np.transpose(H3))))
        # Q2 = eye(N2*k2) + (ep/k2)**2 * (np.kron(D22, np.matmul(np.transpose(H2),H2))) + (ep**2/(k2*k1)) * (np.kron(D21, np.matmul(np.transpose(H3), H3)))
        Qupper = np.concatenate((Q1,csr_matrix((N1*k,N2*k)).toarray()),axis=1)
        Qlower = np.concatenate((csr_matrix((N2*k,N1*k)).toarray(),Q2), axis=1)
        Q = np.concatenate((Qupper,Qlower),axis=0)
        Q = csr_matrix(Q).toarray()
        
        # # calculate maximal value of epsilon that guarantees convergence
        # # construct P (for 2 node types and 3 edges types) for ep = 1
        # ep = 1
        # P11_ = (ep/k1) * (np.kron(W1,H1))
        # P12_ = (ep/k1) * (np.kron(W3,H3))
        # P21_ = (ep/k2) * (np.kron(np.transpose(W3),np.transpose(H3)))
        # P22_ = (ep/k2) * (np.kron(W2,H2))
        # Pupper_ = np.concatenate((P11_,P12_),axis=1)
        # Plower_ = np.concatenate((P21_,P22_), axis=1)
        # P_ = np.concatenate((Pupper_,Plower_),axis=0) # persona-influence matrix P'
        
        # Q1_ = (ep/k1)**2 * (np.kron(D11, np.matmul(H1,np.transpose(H1)))) + (ep**2/(k1*k2)) * (np.kron(D12, np.matmul(H3,np.transpose(H3))))
        # Q2_ = (ep/k2)**2 * (np.kron(D22, np.matmul(H2,np.transpose(H2)))) + (ep**2/(k2*k1)) * (np.kron(D21, np.matmul(H3,np.transpose(H3))))
        # Qupper_ = np.concatenate((Q1_,csr_matrix((N1*k1,N2*k2)).toarray()),axis=1)
        # Qlower_ = np.concatenate((csr_matrix((N2*k2,N1*k1)).toarray(),Q2_), axis=1)
        # Q_ = np.concatenate((Qupper_,Qlower_),axis=0)
        # Q_ = csr_matrix(Q_).toarray()
        
        # P_norm = np.linalg.norm(P_)
        # Q_norm = np.linalg.norm(Q_)
        # eps_convergence = (-P_norm + np.sqrt(P_norm**2 + 4*(Q_norm**2)))/(2*Q_norm)
        # print('recommended to choose ep between:', 0.01*eps_convergence, 'and', 0.1*eps_convergence)
        
        M = P - Q 

        
    else:
        # in case of homogeneous graph: only one W and one H
        # estimate potential matrix H via Linear Compatibility Estimation
        H_total = estimateSingleH(E=E, W=W, k=k, method='LHE')
        H = np.transpose(H_total[0:k,0:k])
        
        # construct persona-influence matrix P
        P = (ep/k) * (np.kron(W,H))
        
        # define diagonal degree matrix D 
        D = np.diag(np.sum(W,1)) # degree matrix for nodes of type 1 connected with other nodes of type 1
        
        # construct Q from D and H 
        Q = (ep/k)**2 * np.kron(H**2,D)
        
        # # calculate maximal value of epsilon that guarantees convergence
        # # construct P (for 2 node types and 3 edges types) for ep = 1
        # ep = 1
        # P11_ = (ep/k1) * (np.kron(W1,H1))
        # P12_ = (ep/k1) * (np.kron(W3,H3))
        # P21_ = (ep/k2) * (np.kron(np.transpose(W3),np.transpose(H3)))
        # P22_ = (ep/k2) * (np.kron(W2,H2))
        # Pupper_ = np.concatenate((P11_,P12_),axis=1)
        # Plower_ = np.concatenate((P21_,P22_), axis=1)
        # P_ = np.concatenate((Pupper_,Plower_),axis=0) # persona-influence matrix P'
        
        # Q1_ = (ep/k1)**2 * (np.kron(D11, np.matmul(H1,np.transpose(H1)))) + (ep**2/(k1*k2)) * (np.kron(D12, np.matmul(H3,np.transpose(H3))))
        # Q2_ = (ep/k2)**2 * (np.kron(D22, np.matmul(H2,np.transpose(H2)))) + (ep**2/(k2*k1)) * (np.kron(D21, np.matmul(H3,np.transpose(H3))))
        # Qupper_ = np.concatenate((Q1_,csr_matrix((N1*k1,N2*k2)).toarray()),axis=1)
        # Qlower_ = np.concatenate((csr_matrix((N2*k2,N1*k1)).toarray(),Q2_), axis=1)
        # Q_ = np.concatenate((Qupper_,Qlower_),axis=0)
        # Q_ = csr_matrix(Q_).toarray()
        
        # P_norm = np.linalg.norm(P_)
        # Q_norm = np.linalg.norm(Q_)
        # eps_convergence = (-P_norm + np.sqrt(P_norm**2 + 4*(Q_norm**2)))/(2*Q_norm)
        # print('recommended to choose ep between:', 0.01*eps_convergence, 'and', 0.1*eps_convergence)
        
        M = P - Q 
    
    #iteratively update final beliefs B
    maxIter = 50
    res = 1; iter=0;
    res_history=[];
    
    while res>1e-8:
        b_old = b
        b = e + np.matmul(M,b_old)
        res = np.sum(np.sum(np.absolute(b_old-b)))
        res_history.append(res)
        iter = iter+1
        if iter == maxIter:
            break
    #print('nr of iterations:', iter)
    
    # extract final beliefs 
    finalBeliefs = np.transpose(np.reshape(b,(k, X.shape[0]), order='F'))
    indices = np.argmax(finalBeliefs, axis=1)
    finals = indices+1    
    
    # get evaluation metrics: accuracy, kappa coefficient and confusion matrix
    acc = accuracy_score(y,finals)
    print('accuracy: ',acc)
    kappa = cohen_kappa_score(y, finals)
    #print('kappa: ', kappa)
    conf_matrix = confusion_matrix(y, finals)
    #print('confusion matrix:', conf_matrix)

    return acc, kappa, conf_matrix, finals


## either load folds from pickle file (uncomment first line), or create them on the spot using the function generateFoldDictionary
#folds = pickle.load(open('folds_'+dataset+'.pickle', 'rb'))
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
        acc, kappa, conf, finals = main_LP(X, y, X_total, k, dimHS, dimLidar, kNN = rs['kNN'], ep = rs['ep'], sig_spe = rs['sig_spe'], sig_spa = rs['sig_spa'],
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
    

