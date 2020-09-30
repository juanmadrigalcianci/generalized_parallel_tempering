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
Nruns=1
#predefines some important parameters
local=0
p=1;#number of parameters
N_temp=4;#number of temperatures
beta=5.**(np.arange(5))
beta_original = np.copy(beta);
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
        return hh
    return (0.5*(h(x_u[:,None]-t_u[None,:],z)+h(x_u[:,None]+t_u[None,:],z))).flatten()

#---------------------------------------------------------------------
#
# Generates Data
#
#---------------------------------------------------------------------

d=np.load('data_cauchy.npy')
sigma=np.load('sig_cauchy.npy')
#---------------------------------------------------------------------
#
# Defines some hyper parameters, Number of temperatures, N samples, etc.
#
#---------------------------------------------------------------------
p=1;#number of parameters
N=int(2.5E4); #number of samples
#lower and upper limit for uniform prior
a=np.array([-5]);
b=np.array([5]);
#Defines likelihood, prior and posterior functions
def L(x):
    return -0.5*np.sum(np.abs(d-u(x))**2)*(dt)/(Nr*sigma**2);
def pr(x):
    return np.log(float(np.all(x>a))*float(np.all(x<b)));
def post(x):
    return L(x)+pr(x)





N_temp=len(beta)
p=1
x0=np.random.random((p,N_temp))
sigma_is=np.array([[0.02],[0.05],[.1],[.5],[2.]])
#sigma_is=np.array([[0.07,0.07],[0.1,0.1],[0.15,0.15],[0.3,0.3]])



mean_r=np.zeros((Nruns,2))
mean_p=np.zeros((Nruns,2))
mean_uw=np.zeros((Nruns,2))
mean_w=np.zeros((Nruns,2))
mean_pf=np.zeros((Nruns,2))
mean_sd=np.zeros((Nruns,2))
mean_y=np.zeros((Nruns,2))

mean_z=np.zeros((Nruns,2))




mx=np.load('true_mean_cauchy.npy')
#my=np.load('true_mean_y_cauchy.npy')

true_th=[mx]
#defines burn in
Burn_in=int(np.ceil(0.2*N))
N=N+Burn_in
#---------------------------------------------------------------------
#
# Runs the algorithm Nrun times and compute the expected value of each run
#
print('started run of ergodic estimator...')
for i in range(Nruns):
    x0=-5+10*np.random.random((p,N_temp))

    print('----------------------------------')
    print('iteration '+str(i))
    
    
    t0=time.time()
    Xsd=base.state_dependent_PT(post,N,beta,sigma_is,x0,Ns=1,Disp=0) #full vanilla, reversible
    print('time PSDPT is '+str(time.time()-t0))
    
    t0=time.time()
    Xptf=base.full_vanilla(post,N,beta,sigma_is,x0,Ns=1,Disp=0) #full vanilla, reversible
    print('time PT is '+str(time.time()-t0))
    
    
    t0=time.time()
    Xuw=base.unweighted_IS(post,N,beta,sigma_is,x0,1,1,Disp=0) #unweighted Is
    print('time UGPT is '+str(time.time()-t0))
    
    t0=time.time()
    Xw,W_IS,W_IS2=base.weighted_IS(post,N,beta_original,sigma_is,x0,Disp=0) #weighted IS
    print('time WGPT is '+str(time.time()-t0))

    t0=time.time()
    Xrwm=base.rwm(post,N*N_temp,[0.5],x0[:,0],Disp=0) #random walk metropolis
    print('time RWM is '+str(time.time()-t0))

    yy,ww=base.weight_samples(Xw,W_IS)
    xx=base.resample_IS(yy,ww,N)
    
    mean_r[i,:]=np.mean(Xrwm[N_temp*Burn_in:],0)
    mean_pf[i,:]=np.mean(Xptf[Burn_in:,:,0],0)
    mean_sd[i,:]=np.mean(Xsd[Burn_in:,:,0],0)

    mean_uw[i,:]=np.mean(Xuw[Burn_in:,:,0],0)
    mean_w[i,:]=np.average(yy[Burn_in*math.factorial(N_temp):,:,0],0,weights=ww[Burn_in*math.factorial(N_temp):])

    mean_y[i,:]=np.mean(xx[Burn_in:,:,0],0)
    
np.save('mean_r_cauchy.npy',mean_r)
np.save('mean_p_cauchy.npy',mean_p)
np.save('mean_uw_cauchy.npy',mean_uw)
np.save('mean_w_cauchy.npy',mean_w)    
np.save('mean_sd_cauchy.npy',mean_sd)    

print('time per iteration')
print(time.time()-t0)    
    
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
