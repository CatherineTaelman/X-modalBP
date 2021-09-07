# -*- coding: utf-8 -*-
"""
LCE for graph with only one node type (homogeneous graph)
"""
import numpy as np
from scipy import optimize
from numpy import linalg as LA
from time import time
import pickle

def define_energy_H(method='LHE',
                    E=None, W=None):
    """Returns an energy function 'energy_H(H)' which is then used in 'estimateH' to find the optimal H
    """

    if method == 'LHE':
        def energy_H(H):
            return LA.norm(E - W.dot(E).dot(H))  #Frobenius norm
    else:
        raise Exception("You specified a non-existing method")
    return energy_H

def transform_hToH(h_vector, k):
    """Transforms a parameter vector for a k dimensional symmetric stochastic matrix into the matrix.
    Allows the optimization problem to become unconstrained.
    """
    if np.isnan(h_vector).any() or (np.abs(h_vector) > 10e10).any():
        print("Problem in 'transform_hToH' input:", h_vector)

    else:
        H = np.zeros((k, k))
        H[0:k, 0:k] = np.transpose(np.reshape(h_vector, (k,k)))

    return H

def estimateSingleH(E, W, k, method='LHE',
              randomize=False, delta=0.1,
              initial_H0=None,
              initial_h0=None,
              constraints=False,
              alpha=0, beta=0,
              verbose=False,
              returnExtraStats = False,
              return_min_energy=False
              ):

    h0 = np.zeros(k*k).dot(1 / k) # initialize h's by all zeros (residual + rows and cols sum to 0)
    energy_H = define_energy_H(W=W, E=E, method='LHE')

    def energy_h(h):
        """changes parameters for energy function from matrix H to free parameters in array"""
        H = transform_hToH(h, k)
        return energy_H(H)


    PRINTINTERMEDIATE = False       # option to print intermediate results from optimizer (for debugging)
    global Nfeval, permutations
    Nfeval = 1
    
    def callbackfct(x):             # print intermediate results, commented out in non-gradient loops below
        global Nfeval
        if PRINTINTERMEDIATE:
            np.set_printoptions(precision=4)
            print('{0:4d}   {2} {1}   '.format(Nfeval, energy_h(x), x))
            print('Iter: {}'.format(Nfeval))

        Nfeval += 1

    
    # define constraints for optimization: rows and cols of individual H's should sum to 0
    def con_sum_rows(h):
        H = transform_hToH(h, k) 
        return [np.sum(H[i,:]) for i in range(len(H))]  

    def con_sum_cols(h):
        H = transform_hToH(h, k) 
        return [np.sum(H[:,i]) for i in range(len(H))] 
    
    cons = [{'type':'eq', 'fun': con_sum_rows},
            {'type':'eq', 'fun': con_sum_cols}]

    #define bound for optimization
    bnds = [(-1,1) for i in range(k*k)]
    
    def optimize_once(h0, energy_h):
        """actual optimization step that can be repeated multiple times from random starting point"""
        result = optimize.minimize(fun=energy_h, x0=h0, method='SLSQP',
                                   bounds=bnds,
                                   callback = callbackfct,
                                   constraints = cons,
                                   options={'disp': False}
                                   )
        h = result.get('x')
        E = result.get('fun')
        return h, E

    # use h0 as starting point
    start = time()
    h, fun = optimize_once(h0, energy_h)
    end = time() - start
    if verbose:
        print("Initial:{} Result:{} Energy:{}".format(np.round(h0, decimals=3), np.round(h, decimals=3), fun))
        print("Time taken by energy optimize: {}".format(str(end)))


    if returnExtraStats:
        return transform_hToH(h, k), end, Nfeval
    elif return_min_energy:
        return h, fun
    else:
        return transform_hToH(h, k)
    
