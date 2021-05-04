#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:29:20 2021

@author: juan
"""



import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
#import tikzplotlib
import glob
import pdb


#obtains the names

MEANS=[]
DIR='res_7/'
EXTENSIONS=['est_pcn_*','est_pt_*','est_sd_*','est_uw_*','est_y_*']
EXPERIMENT=['pCN', 'full PT', 'SDPT', 'UW','w']
Nexp=len(EXPERIMENT)
N_min=np.zeros(Nexp)
dim_q=7
N_runs=np.zeros(len(EXTENSIONS))
means=np.zeros((51,dim_q,Nexp))
for i in range(len(EXTENSIONS)):

    # reads the number of files
    NAMES=glob.glob(DIR+EXTENSIONS[i])
    N_runs[i]=len(NAMES)
    j=0
    for name in NAMES:
        means[j,:,i]=np.load(name)
        j+=1
    



cut=49


for j in range(dim_q): 
    print('------------------------')
    print('------------------------')
    print('------------------------')

    print('QOI '+str(j+1))       
    for i in range(Nexp):
        print(EXPERIMENT[i])
        print('')
        print('mean ')
        print(np.mean(means[:cut,j,i],0))
        
        print('Var')
        print(np.var(means[:cut,j,i],0))
        
    
        print('relv')
        print(np.var(means[:cut,j,0])/np.var(means[:cut,j,i]))
        
        print('------------------------')
            
    
    
    
    
    
