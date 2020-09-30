#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:45:58 2020

@author: juan
"""

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
from pdb import set_trace
import math
#from analyze_chains import analyze_chains
###now that data has been generated, we can do this
Nruns=1
#predefines some important parameters
local=0
p=1;#number of parameters
N=int(1E4); #number of samples
N_temp=5;#number of temperatures
X=np.zeros([N,p,N_temp])#prealloactes
y=np.zeros([N_temp,p]); #preallocates proposals
bt=np.zeros([N,N_temp])
beta=20**(np.arange(4))
beta_original = np.copy(beta);
iscs=np.zeros(N)
acpt=np.zeros(N_temp,); #acceptace rate

signoise=(10000)
disp=1;
dispFreq=100;

#---------------------------------------------------------------------
#
# Generates analytical pde related stuff
#
#---------------------------------------------------------------------
n=100;
x_u = np.arange(-5, 5)
t_u = np.linspace(0, 5, n)
dt=5/n;
Nr=len(x_u)
def u(z):
    
    def h(xx,z):
        hh= np.exp(-100*(xx-z-0.5)**2)+np.exp(-100*(xx-z)**2)+np.exp(-100*(xx-z+0.5)**2)
        #set_trace()
        return hh
    return (0.5*(h(x_u[:,None]-t_u[None,:],z)+h(x_u[:,None]+t_u[None,:],z))).flatten()

#---------------------------------------------------------------------
#
# Generates Data
#
#---------------------------------------------------------------------
# th_true=[-3,3]
# u_true=0.5*(u(th_true[0])+u(th_true[1]))
# sigma=max(u_true)*0.02
# d=u_true+sigma*np.random.standard_normal(np.size(u_true))
# Two subplots, the axes array is 1-d
# np.s
d=np.load('data_cauchy.npy')
sigma=np.load('sig_cauchy.npy')
#---------------------------------------------------------------------
#
# Defines some hyper parameters, Number of temperatures, N samples, etc.
#
#---------------------------------------------------------------------
p=1;#number of parameters
N=int(1E4); #number of samples
#lower and upper limit for uniform prior
a=np.array([-5]);
b=np.array([5]);

beta=2.**(np.arange(4))


#Defines likelihood, prior and posterior functions
def L(x):
    return -0.5*np.sum(np.abs(d-u(x))**2)*(dt)/(Nr*sigma**2);
def pr(x):
    return np.log(float(np.all(x>a))*float(np.all(x<b)));
def post(x):
    return L(x)+pr(x)



import matplotlib.pyplot as plt

Np=200
xs=np.linspace(-5,5,Np)
lp=np.zeros(Np)


for i in range(Np):
    lp[i]=post(xs[i])
import tikzplotlib
    
plt.plot(xs,-lp)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\Phi(\theta;y)$')
tikzplotlib.save("figs/nll.tex")

    
    
    
    
    