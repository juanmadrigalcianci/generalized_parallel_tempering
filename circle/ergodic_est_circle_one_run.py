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
#from analyze_chains import analyze_chains
###now that data has been generated, we can do this
Nruns=100
#predefines some important parameters
local=0
p=2;#number of parameters
N=int(2.5E4); #number of samples
N_temp=4;#number of temperatures
X=np.zeros([N,p,N_temp])#prealloactes
y=np.zeros([N_temp,p]); #preallocates proposals
bt=np.zeros([N,N_temp])
beta=np.array([1,17.1,292.4,5000])
beta_original = np.copy(beta);
iscs=np.zeros(N)
acpt=np.zeros(N_temp,); #acceptace rate

signoise=(10000)

#Defines likelihood, prior and posterior functions
def L(x):
    # circle
    py=-( signoise*( x[0]**2.0+x[1]**2.0 -0.8**2.0 )**2.0)
    return py


#limits to generate first value
amin=np.array([-0.0,-0.0])
amax=np.array([1.0,1.0])

def pr(x):
    return np.log(all(x>amin)*all(x<amax))
def post(x):
    return L(x)+pr(x)







N_temp=len(beta)
p=2
x0=np.random.random((p,N_temp))
sigma_is=np.array([[0.022,0.022],[0.09,0.09],[0.31,0.31],[0.650,0.650]])
#sigma_is=np.array([[0.07,0.07],[0.1,0.1],[0.15,0.15],[0.3,0.3]])



mean_r=np.zeros((Nruns,2))
mean_p=np.zeros((Nruns,2))
mean_uw=np.zeros((Nruns,2))
mean_w=np.zeros((Nruns,2))
mean_pf=np.zeros((Nruns,2))
mean_sd=np.zeros((Nruns,2))

mean_y=np.zeros((Nruns,2))




mx=np.load('true_mean_x_Circle.npy')
my=np.load('true_mean_y_Circle.npy')

true_th=[mx,my]
#defines burn in
Burn_in=int(np.ceil(0.2*N))
N=N+Burn_in
#---------------------------------------------------------------------
#
# Runs the algorithm Nrun times and compute the expected value of each run
#
print('started run of ergodic estimator...')
t0=time.time()
for i in range(Nruns):
    x0=np.random.random((p,N_temp))

    print('----------------------------------')
    print('iteration '+str(i))
    
    Xsd=base.state_dependent_PT(post,N,beta,sigma_is,x0,Ns=1,Disp=0) #full vanilla, reversible

    
    Xptf=base.full_vanilla(post,N,beta,sigma_is,x0,Ns=1,Disp=0) #full vanilla, reversible
    Xuw=base.unweighted_IS(post,N,beta,sigma_is,x0,1,1,Disp=0) #unweighted Is
    Xw,W_IS,W_IS2=base.weighted_IS(post,N,beta_original,sigma_is,x0,Disp=0) #weighted IS
    Xrwm=base.rwm(post,N*N_temp,sigma_is[0],x0[:,0],Disp=0) #random walk metropolis
    yy,ww=base.weight_samples(Xw,W_IS)
    xx=base.resample_IS(yy,ww,N)
    
    mean_r[i,:]=np.mean(Xrwm[N_temp*Burn_in:],0)
    mean_pf[i,:]=np.mean(Xptf[Burn_in:,:,0],0)
    mean_sd[i,:]=np.mean(Xsd[Burn_in:,:,0],0)

    mean_uw[i,:]=np.mean(Xuw[Burn_in:,:,0],0)
    mean_w[i,:]=np.average(yy[Burn_in*math.factorial(N_temp):,:,0],0,weights=ww[Burn_in*math.factorial(N_temp):])

    mean_y[i,:]=np.mean(xx[Burn_in:,:,0],0)
    
    np.save('mean_r_circle_b.npy',mean_r)
    np.save('mean_p_circle_b.npy',mean_p)
    np.save('mean_uw_circle_b.npy',mean_uw)
    np.save('mean_w_circle_b.npy',mean_w)    
    np.save('mean_sd_circle_b.npy',mean_sd)    
    
    
print('----------------------------------')
print('----------------------------------')
print('means')


print(np.mean(mean_r,0))
print(np.mean(mean_pf,0))
print(np.mean(mean_sd,0))



print(' GPT ')

print(np.mean(mean_uw,0))
print(np.mean(mean_w,0))
print(np.mean(mean_y,0))


print('----------------------------------')
print('variances')
print(np.var(mean_r,0))
print(np.var(mean_pf,0))
print(np.var(mean_sd,0))

print(' GPT ')

print(np.var(mean_uw,0))
print(np.var(mean_w,0))
print(np.var(mean_y,0))


print('----------------------------------')
print('bias ^2 ')
print(np.abs((np.mean(mean_r,0)-true_th))**2.0)
print(np.abs((np.mean(mean_pf,0)-true_th))**2.0)
print(np.abs((np.mean(mean_sd,0)-true_th))**2.0)

print(' GPT ')

print(np.abs((np.mean(mean_uw,0)-true_th))**2.0)
print((np.abs(np.mean(mean_w,0)-true_th))**2.0)
print((np.abs(np.mean(mean_y,0)-true_th))**2.0)

print('----------------------------------')
print('MSE ')
print(np.var(mean_r,0)+(np.mean(mean_r,0)-true_th)**2.0)
print(np.var(mean_pf,0)+(np.mean(mean_pf,0)-true_th)**2.0)
print(np.var(mean_sd,0)+(np.mean(mean_sd,0)-true_th)**2.0)

print(' GPT ')
print(np.var(mean_uw,0)+(np.mean(mean_uw,0)-true_th)**2.0)
print(np.var(mean_w,0)+(np.mean(mean_w,0)-true_th)**2.0)
print(np.var(mean_y,0)+(np.mean(mean_y,0)-true_th)**2.0)
 #%%
yy,ww=base.weight_samples(Xw,W_IS)

xx=base.resample_IS(yy,ww,N)

import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interpn
#from load_emcee import get_samples_emcee
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


def density_scatter( x , y, ax = None, sort = True, bins = 20, w=None, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins,weights=w,range=[[0,1],[0,1]])
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z,s=2, **kwargs )
    return ax

BI=5000
Max=25000

fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (9,6))

axes[0,0]=density_scatter( Xrwm[BI:4*Max,0], Xrwm[BI:4*Max,1], ax=axes[0,0], bins = [100,100], cmap='Spectral')
axes[0,0].set_title(r'RWM')
axes[0,0].set_xlim(0,1)
axes[0,0].set_ylim(0,1)


axes[0,1]=density_scatter( Xptf[BI:Max,0,0], Xptf[BI:Max,1,0],ax=axes[0,1], bins = [100,100], cmap='Spectral')
axes[0,1].set_title(r'PT')
axes[0,1].set_xlim(0,1)
axes[0,1].set_ylim(0,1)


axes[0,2]=density_scatter( Xsd[BI:Max,0,0], Xsd[BI:Max,1,0],ax=axes[0,2], bins = [100,100], cmap='Spectral')
axes[0,2].set_title(r'PSDPT')
axes[0,2].set_xlim(0,1)
axes[0,2].set_ylim(0,1)

axes[1,0]=density_scatter( Xuw[BI:Max,0,0], Xuw[BI:Max,1,0],ax=axes[1,0],bins = [100,100], cmap='Spectral')
axes[1,0].set_title(r'UGPT')
axes[1,0].set_xlim(0,1)
axes[1,0].set_ylim(0,1)

axes[1,1]=density_scatter( xx[BI:Max,0,0], xx[BI:Max,1,0] , ax=axes[1,1],bins = [100,100], cmap='Spectral')
axes[1,1].set_title(r'WGPT')
axes[1,1].set_xlim(0,1)
axes[1,1].set_ylim(0,1)

axes[1,2]=density_scatter( Xw[BI:Max,0,0], Xw[BI:Max,1,0],ax=axes[1,2],bins = [100,100], cmap='Spectral')
axes[1,2].set_title(r'WGPT (inv)')
axes[1,2].set_xlim(0,1)
axes[1,2].set_ylim(0,1)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])