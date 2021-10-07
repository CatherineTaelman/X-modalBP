# -*- coding: utf-8 -*-
"""
Graph-based label propagation method that supports heterogeneous data, based on ZooBP propagation rules (Eswaran et al., 2016)

PARAMETERS for label_propagation:

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
             
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.metrics import pairwise
from LCE import estimateH
from LCE_singleH import estimateSingleH
from numpy import linalg as LA
from sklearn.metrics import cohen_kappa_score
import time
from scipy import sparse
from memory_profiler import profile
from scipy.sparse import diags
from sys import getsizeof

  
# @profile
def label_propagation(X, y, X_total, k, dimHS, dimLidar, kNN, ep, sig_spe, sig_spa, r, choice, fraction=0.8,):
    assert len(X) == len(y), 'X and y must have same length'
    
    verbose=False # Set to True for time stamps

#%% Split data coords vs rest. Define GT
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
#%%     If statement    
    if choice != 'full overlap':
        gt_adjusted_combi = gt_adjusted[0:split] #ground truth for combi data (overlapping region)
        gt_adjusted_nonOverlap = gt_adjusted[split:] # ground truth for lidar data OR for HS data in non-overlapping region
        # construct weigted adjacency matrices W1, W2 and W3 
        # W1 and W2 are constructed using spectral-spatial similarity metric of Shi&Malik

#%%     Init W1             
        S = len(X_overlap)     
        W1=lil_matrix((S,S))
        start = time.time()
        for i in range(0,S):
            normspa = np.sum(np.abs(X_overlap_coords[i] - X_overlap_coords[i:])**2,axis=1)**(1./2)
            idx=np.where(normspa<r)[0]+i
            
            normspe = np.sum(np.abs(X_overlap[i] - X_overlap[idx,:])**2,axis=1)**(1./2)
            
            w_temp=np.exp(-(normspe**2/(sig_spe**2)))*np.exp(-(normspa[idx-i]**2/(sig_spa**2)))
            W1[i,idx]=w_temp

        W1=W1.tocsr()
        W1=W1+W1.transpose()-diags(W1.diagonal())
        
        if verbose==True:
            print("Method took: {:.3f}".format(time.time()-start) + " seconds for initialization of W1")
            print("")                
#%%     Init W2 Non overlap
        start = time.time()
        S = len(X)-split
        W2=lil_matrix((S,S))
        
        for i in range(0,S):
            if choice == 'hyperspectral':
                normspa = np.sum(np.abs(nonOverlap_hyper_coords[i] - nonOverlap_hyper_coords[i:])**2,axis=1)**(1./2)
                idx=np.where(normspa<r)[0]+i
                normspe = np.sum(np.abs(X_hyper[i] - X_hyper[idx,:])**2,axis=1)**(1./2)
            elif choice == 'lidar': 
                normspa = np.sum(np.abs(nonOverlap_lidar_coords[i] - nonOverlap_lidar_coords[i:])**2,axis=1)**(1./2)
                idx=np.where(normspa<r)[0]+i
                normspe = np.sum(np.abs(X_lidar[i] - X_lidar[idx,:])**2,axis=1)**(1./2)
                

            w_temp=np.exp(-(normspe**2/(sig_spe**2)))*np.exp(-(normspa[idx-i]**2/(sig_spa**2)))
            W2[i,idx]=w_temp
       
        W2=W2.tocsr()
        W2=W2+W2.transpose()-diags(W2.diagonal())
        
        if verbose==True:
            print("Method took: {:.3f}".format(time.time()-start) + " seconds for initialization of W2")
            print("")
#%%     Init W3 Non overlap:
        # W3 is constructed using Gaussian kernel (spectral similarity) and kNN
        start=time.time()
        metric = pairwise.rbf_kernel(X_overlap, nonOverlap_padded, gamma=sig_spe) # Gaussian kernel
        sim = np.argsort(-metric, axis=1)
        mask = np.zeros(metric.shape)
        ind = sim[:,:kNN] # indices of k-nearest neighbours
        for mask_row, ind_row in zip(mask, ind):
            mask_row[ind_row] = 1
        W3 = metric.copy()
        W3[mask==0] = 0
        W3=csr_matrix(W3)
        
        if verbose==True:
            print("Method took: {:.3f}".format(time.time()-start) + " seconds for initialization of W3")
            print("------------------------------------------------------------------------")
        
        gtTypes= [gt_adjusted_combi, gt_adjusted_nonOverlap]
        classes = [k, k]
        
        N1 = gt_adjusted_combi.shape[0] # amount of nodes of type 1
        N2 = gt_adjusted_nonOverlap.shape[0] # amount of nodes of type 2
        nodes = [N1,N2]          
#%%     If statement     
    else:

#%%     Init W full overlap
        S = len(X_overlap)           
        W=lil_matrix((S,S))
        start = time.time()
        for i in range(0,S):
            normspa = np.sum(np.abs(X_overlap_coords[i] - X_overlap_coords[i:])**2,axis=1)**(1./2)
            idx=np.where(normspa<r)[0]+i
            
            normspe = np.sum(np.abs(X_overlap[i] - X_overlap[idx,:])**2,axis=1)**(1./2)
            
            w_temp=np.exp(-(normspe**2/(sig_spe**2)))*np.exp(-(normspa[idx-i]**2/(sig_spa**2)))
            W[i,idx]=w_temp
            

        W=W.tocsr()
        W=W+W.transpose()-diags(W.diagonal())
        if verbose==True:
            print("Method took: {:.3f}".format(time.time()-start) + " seconds for initialization of W")
            print("------------------------------------------------------------------------")
            
        gtTypes= [gt_adjusted] 
        classes = [k , k]
        
        N = gt_adjusted.shape[0] # amount of nodes
        nodes = [N]
#%%     Construct prior belief and initialize final belief:     
    # construct prior belief vector e: 3 steps
    E_total=[]
    for t, c in zip(gtTypes, classes):
        # step 1: reshape priors per node type into format (n_s x k_s) with 1 indicating it belongs to that class
        E = np.zeros([len(t), c])
        for i in range(len(t)):
            if t[i] != 0:
                pos = int(t[i] - 1)
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
 
#%%     If statement
    if choice != 'full overlap':

#%%     Calculate E, W, H, P and Q for non overlap
        E1 = E_total[0]
        E2 = E_total[1]
        
        Eupper = np.concatenate((E1,np.zeros_like(E1)),axis=1)
        Elower = np.concatenate((np.zeros_like(E2),E2),axis=1)
        E_tot = np.concatenate((Eupper,Elower),axis=0)
        
        Wupper=sparse.hstack([W1,W3])
        Wlower=sparse.hstack([W2,W3.transpose()])
        W_tot=sparse.vstack([Wupper,Wlower])

        
        if verbose==True:
            print("W_tot took {:.3f}".format(time.time()-start)+" seconds to initialize")
            print("")
            
        # estimate potential matrices H via Linear Compatibility Estimation (LCE)
        # H1: combi-combi, H2: Lidar-lidar, H3: combi-lidar edges
        
        start=time.time()
        
        H_total = csr_matrix(estimateH(E=csr_matrix(E_tot), W=W_tot, k=k, method='LHE'))  
        H1 = H_total.transpose()[0:k,0:k]
        H2 = H_total[k:,k:]
        H3 = H_total[0:k,k:]

        if verbose==True:
            print("EstimateH took {:.3f}".format(time.time()-start)+" seconds")
            print("")
        
        # construct P (for 2 node types and 3 edges types)
        start=time.time()
        
        D11 = csr_matrix(diags(np.asarray(np.sum(W1,1)).squeeze()))
        D22 = csr_matrix(diags(np.asarray(np.sum(W2,1)).squeeze()))
        D12 = csr_matrix(diags(np.asarray(np.sum(W3,1)).squeeze()))
        D21 = csr_matrix(diags(np.asarray(np.sum(W3,0)).squeeze()))
        if verbose==True:
            print("D {:.3f}".format(time.time()-start)+" seconds")
        # construct Q from D's and H's 
        
        start=time.time()
        
        Q1 = (ep/k)**2 * sparse.kron(D11, H1@H1.transpose()) + (ep**2/(k*k) * sparse.kron(D12,H3@H3.transpose()))
        Q2 = (ep/k)**2 * sparse.kron(D22, H2@H2.transpose()) + (ep**2/(k*k) * sparse.kron(D21,H3.transpose()@H3))
        
        Qupper=sparse.hstack([Q1,csr_matrix((N1*k,N2*k))])
        Qlower=sparse.hstack([csr_matrix((N2*k,N1*k)),Q2])
        Q = sparse.vstack([Qupper,Qlower])
        Q = Q.tocsr()
        if verbose==True:
            print("Q {:.3f}".format(time.time()-start)+" seconds")
        
        start=time.time()
        
        P11 = (ep/k) * (sparse.kron(W1,H1))
        P12 = (ep/k) * (sparse.kron(W3,H3))
        P21 = (ep/k) * (sparse.kron(W3.transpose(),H3.transpose()))
        P22 = (ep/k) * (sparse.kron(W2,H2))
        
        Pupper = sparse.hstack([P11,P12])
        del P11,P12
        Plower = sparse.hstack([P21,P22])
        del P21,P22
        M = sparse.vstack([Pupper,Plower])-Q
        M = M.tocsr()
        
        if verbose==True:
            print("sparsekron took {:.3f}".format(time.time()-start)+" seconds")
            print("------------------------------------------------------------------------")
            
        
        
        # start=time.time()
        # M = P - Q
        # if verbose==True:
        #     print("P-Q:{:.3f}".format(time.time()-start)+" seconds")
        #     print("")
#%%     If statement
    else:
        
#%%     Calculate E, W, H, P and Q for full overlap      
        # in case of homogeneous graph: only one W and one H
        # estimate potential matrix H via Linear Compatibility Estimation
        start=time.time()
        
        H_total = estimateSingleH(E=csr_matrix(E), W=W, k=k, method='LHE')
        H = csr_matrix(np.transpose(H_total[0:k,0:k]))
        
        if verbose==True:
            print("EstimateH took {:.3f}".format(time.time()-start)+" seconds")
            print("")


        # construct persona-influence matrix P

        
        # define diagonal degree matrix D 
        start=time.time()
        D = csr_matrix(diags(np.asarray(np.sum(W,1)).squeeze())) # degree matrix for nodes of type 1 connected with other nodes of type 1
        if verbose==True:
            print("D {:.3f}".format(time.time()-start)+" seconds")
        
        # construct Q from D and H
        start=time.time()
        Q=(ep/k)**2*sparse.kron(H**2,D)
        Q=Q.tocsr()
        if verbose==True:
            print("Q {:.3f}".format(time.time()-start)+" seconds")
        
        start=time.time()
        
        M = (ep/k)*sparse.kron(W,H)-Q
        M = M.tocsr()
        if verbose==True:
            print("sparsekron took {:.3f}".format(time.time()-start)+" seconds")   
            print("------------------------------------------------------------------------")

        
        # start=time.time()
        # M = P - Q 
        # if verbose==True:
        #     print("P-Q:{:.3f}".format(time.time()-start)+" seconds")
        #     print("")
#%%     Solve for B     
        #iteratively update final beliefs B
    maxIter = 50
    res = 1; iter=0;
    res_history=[];
    e=csr_matrix(e)
    while res>1e-8:
        b_old = b
        
        b=e+M@b_old
        res = np.sum(np.absolute(b_old-b))
        res_history.append(res)
        iter = iter+1
        if iter == maxIter:
            break
        # print('nr of iterations:', iter)
    
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