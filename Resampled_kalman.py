#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:45:22 2017

@author: jan
"""

#import all necessary packages
import numpy as np
import pandas as pd
import re
#import matplotlib.pyplot as plt
from numpy import dot
from scipy.linalg import inv
from multiprocessing import Pool
import time
from collections import defaultdict

def resample(_data, p, seed=None):
    """
    Performs a stationary block bootstrap resampling of elements from a time 
    series, or rows of an input matrix representing a multivariate time series
    
    Inputs:
        data - An MxN numerical array of data to be resampled. It is assumed
               that rows correspond to measurement observations and columns to 
           items being measured. 1D arrays are allowed.
        p    - A scalar float in (0,1] defining the parameter for the geometric
               distribution used to draw block lengths. The expected block 
           length is then 1/p                

    Keywords:
        seed - A scalar integer defining the seed to use for the underlying
       random number generator.
    
    Return:
        A three element list containing
    - A resampled version, or "replicate" of data
    - A length M array of integers serving as indices into the rows
      of data that were resampled, where M is the number of rows in 
      data. Thus, if indices[i] = k then row i of the replicate data 
      contains row k of the original data.
    - A dictionary containing a mapping between indices into data and
      indices into the replicate data. Specifically, the keys are the
      indices for unique numbers in data and the associated dict values 
      are the indices into the replicate data where the numbers are.

    Example:            
        In [1]: import numpy as np
        In [2]: x = np.random.randint(0,20, 10)
        In [3]: import stationary_block_bootstrap as sbb
        In [4]: x_star, x_indices, x_indice_dict = sbb.resample(x, 0.333333)
        In [5]: x
        Out[5]: array([19,  2,  9,  9,  9, 10,  2,  2,  0, 11])
        In [6]: x_star
        Out[6]: array([19, 11,  2,  0, 11, 19,  2,  2, 19,  2])
        In [7]: x_indices
        Out[7]: array([0, 9, 7, 8, 9, 0, 6, 7, 0, 1])

        So, x_indices[1] = 9 means that the 1th element of x_star corresponds 
        to the 9th element of x
        
        In [8]: x_indice_dict
        Out[8]: {0: [0, 5, 8], 7: [2, 6, 7, 9], 8: [3], 9: [1, 4]}
    
        So, x[0] = 19 occurs in position 0, 5, and 8 in x_star. Likewise, 
        x[9] = 11 occurs in positions 1 and 4 in x_star
    
"""
    num_obs = np.shape(_data)[0]
    num_dims = np.ndim(_data)
    assert num_dims == 1 or num_dims == 2, "Input data must be a 1 or 2D array"
    #There is a probably smarter way to wrap the series without doubling
    #the data in memory; the approach below is easy but wasteful
    if num_dims == 1:
        wrapped_data = np.concatenate((_data, _data)) 
    elif num_dims == 2:
        wrapped_data = np.row_stack((_data, _data)) 
    
    assert p > 0 and p <=1, "p must be in (0,1]"
    
    if seed is not None:
        np.random.seed(seed=seed)

    #Make the random variables used in the resampling ahead of time. Could be
    #problematic for memory if num_obs is huge, but doing it this way cuts down 
    #on the function calls in the for loop below...
    choices = np.random.randint(0, num_obs, num_obs)
    unifs = np.random.uniform(0, 1, num_obs)
    
    #Let x and x* be length-n vectors with x*[0] = x[0]. 
    #Then x*[1] = x[1] with probability 1-p. With probability p, x*[1] will
    #equal a random i for x[i]. The expected run length is 1/p = "block length"
    indices = -np.ones(num_obs, dtype=int)
    indices[0] = 0
        
    for i in range(1, num_obs):
        if (unifs[i] > p): 
            indices[i] = indices[i-1] + 1 
        else:
            indices[i] = choices[i]

    if num_dims == 1:        
        resampled_data = wrapped_data[indices]   
        index_to_data_map = dict((x, i) for i, x in enumerate(wrapped_data))
        bootstrap_indices = map(index_to_data_map.get, resampled_data)
    elif num_dims == 2:
        #Mapping is the same for each column with respect to which rows are
        #resampled, so just consider one variable when mapping indices to data...
        resampled_data = wrapped_data[indices, :]   
        index_to_data_map = dict((x, i) for i, x in enumerate(wrapped_data[:,0]))
        bootstrap_indices = map(index_to_data_map.get, resampled_data[:,0])
        
    #bootstrap_indices = [index % num_obs for index in bootstrap_indices]
    
    #The code below is making a dictionary mapping of observations resampled to
    #where in the array of indices that observation shows up. Some observations
    #are resampled multiple times, others not at all, in any given replicate
    #data set. The straight-forward code is
    # try:
    #   items = dict[key]
    # except KeyError:
    #   dict[key] = items = [ ]
    #   items.append(value)
    """
    index_occurences = defaultdict(list)
    for pos, index in enumerate(bootstrap_indices):
        index_occurences[index].append(pos)
    
    index_dict = dict(index_occurences)
    """
    #Need to make the indices we save be bounded by 0:num_obs. For example, 
    #data[0,:] = data[num_obs,:]  and data[1,:] = data[num_obs+1,*] etc     
    #because we wrapped the data. But, with respect to the data arrays used 
    #elsewhere, an index of num_obs+1 is out of bounds, so num_obs should be 
    #converted to 0, num_obs+1 to 1, etc...   

    return [resampled_data, indices % num_obs]#, index_dict] 
    #end resample() 

train = pd.read_csv("train_1.csv").fillna(0)
keys = pd.read_csv("key_1.csv")
ssm = pd.read_csv("sample_submission_1.csv")

reg_keys= re.compile(r'(.*)_\d{4}-\d{2}-\d{2}')

page_keys = keys["Page"]
page_keys_nodate = [reg_keys.findall(page_keys[idx])[0] for idx in range(len(page_keys))]

d = defaultdict(int)
for i in page_keys_nodate:
    d[i] += 1
    
train_set = d.keys()
train.index = train["Page"]
train = train.drop("Page",axis = 1)
train = train.reindex(train_set)

val_x = [0.,0.]
val_P =[1,1.]
val_H =[1.,1.]
Q_var = 10
val_R = 10
dim = 2
n_sam = 20
dt = 1
all_xs = []
all_Ps = []
pred = []
pred_len = ssm.shape[0]/train.shape[0]
row = train.iloc[0]
data = pd.concat([pd.Series(row),row-pd.Series(row).shift(1)],axis=1)[1:]
n_days = data.shape[0]
result = []

def resampled_kalman(count):
    dt = 1.0
    out = []
    #print(count)
    np.random.seed()
    x = np.array([val_x]).T #2*1
    P = np.diag(val_P) #2*2
    F = np.array([[1.,dt], 
                  [0,1]]) # 2*2
    H = np.diag(val_H) #2*2
    indexes = np.sort(resample(range(n_days),0.5)[0])
    #print(indexes)
    for idx,z in enumerate(indexes):
        z = data.iloc[z]
        #if z[0]<0:
        #    z[0] = 0.0000001
        z = np.array([[z[0]],
                     [z[1]]]) # 2*1
        # predict
        x = dot(F, x) # 2*2 * 2* 1
        Q = np.random.normal(loc=0,scale = Q_var,size =(dim,dim))
        P = dot(F, P).dot(F.T) + Q # 2*2 * 2*2 * 2*2
        #update
        R = np.diag([val_R,val_R]) # 2*2
        S = dot(H, P).dot(H.T) + R # 2*2 * 2*2 * 2*2
        K = dot(P, H.T).dot(inv(S)) # 2*2 * 2*2 * 2*2
        y = z - dot(H, x) # 2*1 - 2*2 * 2*1
        #print(z)
        x += dot(K, y) # 2*1 + 2*2 * 2*1
        P = P - dot(K, H).dot(P) # 2*2 - 2*2 * 2*2 * 2*2
    for idx in range(pred_len):
        x = dot(F, x)
        Q = np.random.normal(loc=0,scale = Q_var,size =(dim,dim))
        P = dot(F, P).dot(F.T) + Q
        #print(x)
        out.append(x[0])
    return(out)
#start_time = time.time()    
num_threads = 10

for n_rows in range(train.shape[0]):
    row = train.iloc[n_rows]
    data = pd.concat([pd.Series(row),row-pd.Series(row).shift(1)],axis=1)[1:]
    #xs, cov = np.zeros((n_sam,data.shape[0])),[]
    #xs, cov = [],[]
    pool = Pool(processes = num_threads)                         # Create a multiprocessing Pool
    result = pool.map(resampled_kalman, range(n_sam))# proces data_inputs iterable with pool
    xs = np.nanmedian(result,axis = 0)
    pool.close()
    pool.join()
   # print(xs_.shape)
   #pred.append(xs_)
    #print(pred)
    if n_rows % 100 == 0:
        print(n_rows)
        #print(xs)
    pred.append(xs) 
#elapsed_time = time.time() - start_time    
#print(elapsed_time)

ssm.iloc[:,1] = np.ravel(pred)    
ssm.to_csv("prediction.csv",index = False)