#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:21:56 2020

@author: juan
"""

import numpy as np
import base as base
import time
import math
import acoustic_driver_two_srource as ac
import matplotlib.pyplot as plt
#------------------------   parallel libraries     -------------------
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#name=MPI.Get_processor_name()
#---------------------------------------------------------------------
##### testes the codes
    

# defines parameters

a=np.array([0.,0.,8, 4800] )
b=np.array([3.,2.,12,5200])
p=len(a)
Nruns=1

#---------------------------------------------------------------------
#
# Generates data
#---------------------------------------------------------------------

#generates data
th_true=np.array([1.5,1,10,5000])

y_data=np.load('y_true_data.npy')
sigma=np.load('sigma_ac.npy')
tt=np.load('tt.npy')


#---------------------------------------------------------------------
#
# Defines posterior
#---------------------------------------------------------------------

def post(x):
    
    lp=ac.posterior(x, y_data, tt, a, b,sigma)
    
    return lp

e=1e-10
xx=np.linspace(0+e,3.0-e,101)
yy=np.linspace(0+e,2.0-e,101)

z=np.zeros((len(xx),len(yy)))
iteration=0
for i in range(len(xx)):
    for j in range(len(yy)):
        theta=np.array((xx[i],yy[j],th_true[2],th_true[3]))
        z[i,j]=post(theta)
        print('-----------------')
        print('iteration '+str(iteration))
        print('x ='+str(round(xx[i],3))+' y='+str(round(yy[j],3))+ ' -log_post='+str(round(z[i,j],3)))
        iteration+=1
    
np.save('nll_ac.npy',z)


#%%
plt.figure(figsize=(9,4))

X=np.meshgrid(xx)[0]
Y=np.meshgrid(yy)[0]
from matplotlib import rc
import tikzplotlib
plt.style.use('ggplot')
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
     'size' : '12'})
rc('text', usetex=True)
rc('lines', linewidth=2)
plt.rcParams['axes.facecolor']='w'
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']


p=plt.contourf(X,Y,z.T,100,cmap='Spectral')
plt.colorbar(p)
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')

plt.plot([1.0],[1.98],'*',color='m')
plt.plot([1.5],[1.98],'*',color='m')
plt.plot([2.0],[1.98],'*',color='m')

plt.plot([1.5],[1.0],'*',color='k')

plt.savefig('nll_ac.png',format='png',bbox_inches='tight')
tikzplotlib.save("nll_ac.tex")


plt.show()
