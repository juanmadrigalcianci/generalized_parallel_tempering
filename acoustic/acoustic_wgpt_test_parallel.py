#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:36:37 2019

@author: jmadriga
"""

#---------------------------------------------------------------------
import numpy as np
import base as base
import time
import math
import acoustic_driver_two_srource as ac
import sys

id_proc=1#int(sys.argv[1]) uncomment for cluster run 
print('id proc is '+str(id_proc))
##### testes the codes
    

# defines parameters

a=np.array([0.,0.,5, 4000] )
b=np.array([3.,2.,25,6000])
p=len(a)

#---------------------------------------------------------------------
#
# Generates data
#---------------------------------------------------------------------

# # #generates data
th_true=np.array([1.5,1,10,5000])
y_data=np.load('y_true_data.npy')
sigma=np.load('sigma_ac.npy')
tt=np.load('tt.npy')


#---------------------------------------------------------------------
#
# Defines posterior
#---------------------------------------------------------------------

def post(x):
    #pdb.set_trace()
    lp=ac.posterior(x, y_data, tt, a, b,sigma)
    
    return lp

#---------------------------------------------------------------------
#
# Defines some hyper parameters, Number of temperatures, N samples, etc.
#
#---------------------------------------------------------------------
N=int(0.7E4); #number of samples
N_temp=4;#number of temperatures
Ns=1; #How often do we swap
X=np.zeros([N,p,N_temp])#prealloactes
y=np.zeros([N_temp,p]); #preallocates proposals
bt=np.zeros([N,N_temp])
T0=400.0**(1.0/3.0)
beta = np.array(T0**np.arange(0,N_temp));
beta_original = np.copy(beta);
sigma_rwm=np.array([0.01,0.01,0.2,5]);#*linspace(1,100,N_temp);

sigma_is=np.array([[0.01,0.01,0.2,5],[0.06,0.06,0.4,14],[0.3,0.3,0.6,20], \
                   [1.,1.,1,50]]);#*linspace(1,100,N_temp);





Burn_in=int(np.ceil(0.2*N))
N=N+Burn_in
#---------------------------------------------------------------------
#
# Runs the algorithm Nrun times and compute the expected value of each run
#
print('started run of ergodic estimator...')

t0=time.time()

    
np.random.seed(id_proc+2)
x0=a+(b-a)*np.random.random((p,N_temp))
 

print('----------------------------------')

Xw,W_IS,W_IS2=base.weighted_IS(post,N,beta_original,sigma_is,x0.T,Disp=1) #weighted IS
yy,ww=base.weight_samples(Xw,W_IS)
mean_w=np.average(yy[Burn_in*math.factorial(N_temp):,:,0],0,weights=ww[Burn_in*math.factorial(N_temp):])



np.save('mean_w_acoustic_'+str(id_proc).zfill(3)+'.npy',mean_w)      

    
print('Done!')    
    
    
    





