#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:58:35 2021

@author: juan
"""
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import time
from forward_wave import forward
from wave_generate_data import th_true
from base import compute_ratio_uw
import itertools as it


K=np.arange(1,12)
field=th_true()
t0=time.time()
A=forward(field)
tA=time.time()-t0

tB=0.0011


time_PDE=np.zeros(len(K))
time_PDE2=np.zeros(len(K))

time_PERM=np.zeros(len(K))
time_W=np.zeros(len(K))
time_TOTAL=np.zeros(len(K))
time_TOTAL2=np.zeros(len(K))



for i in K:
    time_PDE[i-1]=i*tA   
    time_PDE2[i-1]=i*tB   

    t0=time.time()
    beta=np.random.random(i)
    compute_ratio_uw(beta,beta)
    tf=time.time()-t0
    time_W[i-1]=tf
    
    t0=time.time()
    per_beta=list(it.permutations(beta))
    tf=time.time()-t0
    time_PERM[i-1]=tf
    
    
    time_TOTAL[i-1]=time_PDE[i-1]+time_W[i-1]
    time_TOTAL2[i-1]=time_PDE2[i-1]+time_W[i-1]

#%%
plt.semilogy(K,time_PDE,label='Wave')
plt.semilogy(K,time_PDE2,label='Cauchy')

plt.semilogy(K,time_W,label='Permutation')
#plt.semilogy(K,time_W,label='W')
plt.semilogy(K,time_TOTAL,label='Total Wave')
plt.semilogy(K,time_TOTAL2,label='Total Cauchy')


plt.grid(True)
plt.legend()

tikzplotlib.save('cost_comparisson.tex')


