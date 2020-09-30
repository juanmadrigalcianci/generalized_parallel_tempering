#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:07:56 2019

@author: jmadriga
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
import time
from scipy.sparse.linalg import spsolve
import scipy
from matplotlib import rc

#
#   imports graphical paramters
#
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




t=0
T=[1,17.1,292.4,5000]
fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (12,2.7))
for ax in axes:
    
    #Defines likelihood, prior and posterior functions
    def L(x,T=1):
        # circle
        py=-( 10000*( x[0]**2.0+x[1]**2.0 -0.8**2.0 )**2.0)
        return py
    
    
    #limits to generate first value
    amin=np.array([-0.0,-0.0])
    amax=np.array([2.0,2.0])
    
    def pr(x):
        return np.log(all(x>amin)*all(x<amax))
    def post(x):
        return L(x)+pr(x)
    
    def normal(x,y,T=1):
        return  np.exp(post([x,y])/T)
    
    N=500
    
    am=-0.0
    bm=1.0
    x=np.linspace(am,bm,N)
    z=np.zeros((N,N))
    my=0.0
    mx=0.0
    
    dh=(bm-am)/N
    for i in range(N):
        for j in range(N):
            z[i,j]=normal(x[i],x[j],T[t])
            my+=x[j]*z[i,j]
            mx+=x[i]*z[i,j]
            
    Z=np.sum(z*dh**2)                        
    Mx=(mx*dh**2)/Z
    My=(my*dh**2)/Z
    X,Y=np.meshgrid(x,x)
    p=ax.contourf(X,Y,z.T/Z,100, cmap='Spectral',xlabel='$\\theta_1$',ylabel='$\theta_2$')
    ax.set_title(r'$\mu_{'+str(t+1)+'},$  $T_'+str(t+1)+'='+str(T[t])+'$')
    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$\\theta_2$')
    plt.tight_layout(True)
    fig.colorbar(p, ax=ax)
    #ax.set_colorbar(p)
    t=t+1
plt.savefig('true_cirlce_density_'+str(t)+'.png', format='png',bbox_inches='tight', transparent=True)
plt.show()

