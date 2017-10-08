# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:23:20 2017

@author: soroosh
"""
import numpy as np
from scipy import signal
from sklearn import svm, pipeline, base, metrics
import eegtools
 
def CSP_train(sig_1, sig_2):
    # The input matrices should be Nsample X Nch X Ntrials 
    
    #(b, a) = signal.butter(3, np.array([8, 30]) / (d.sample_rate / 2), 'band')
    (b, a) = signal.butter(3, np.array([8, 30]) / (256. / 2), 'band')
    sig_1 = signal.lfilter(b, a, sig_1, 1)
    sig_2 = signal.lfilter(b, a, sig_2, 1)
    
    cov_1 = []
    cov_2 = []
    for i in range(np.shape(sig_1)[2]):  # the loop over trials
        cov_1.append(np.cov(sig_1[:,:,i]))
        
    cov_1 = np.asarray(cov_1)
        
    for i in range(np.shape(sig_2)[2]):
        cov_2.append(np.cov(sig_2[:,:,i]))

    cov_2 = np.asarray(cov_2)
        
    mean_cov_1 = np.mean(cov_1, axis = 0)     
    mean_cov_2 = np.mean(cov_2, axis = 0)
    # -----------------------------------------------------------------------------
    
    W = eegtools.spatfilt.csp(mean_cov_1, mean_cov_2, 4)
    
    # -----------------------------------------------------------------------------
    for i in range(np.shape(sig_1)[2]):
        try:
            feature_1 = np.vstack(feature_1 , np.dot(W,sig_1[:,:,i]))
        except NameError:
            feature_1 = np.dot(W,sig_1[:,:,i])
        
    for i in range(np.range(sig_2)[2]):
        try:
            feature_2 = np.stack(feature_2 , np.dot(W, sig_2[:,:,i]))
        except NameError:
            feature_2 = np.stack(feature_2 , np.dot(W, sig_2[:,:,i]))
    
    return feature_1, feature_2, W


def CSP_test(sig, W):
     # sig should be like : Nsample X Nch
     
   # (b, a) = signal.butter(3, np.array([8, 30]) / (d.sample_rate / 2), 'band')
    (b, a) = signal.butter(3, np.array([8, 30]) / (256 / 2), 'band')
    sig = signal.lfilter(b, a, sig, 1)
    feature = np.dot(W, sig)
    
    return feature
