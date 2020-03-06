#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:36:37 2019

@author: jmadriga
"""

#---------------------------------------------------------------------
import numpy as np
import base as base
from scipy.sparse import spdiags
import time
from scipy.sparse.linalg import spsolve
import scipy
# Number of runs 
Nruns=10


def compute_rmse(x,xt):
    N=np.shape(x,0)
    m=np.mean(x)
    bias2=(m-xt)**2.0
    varN=np.var(x,0)/N
    
    return  np.sqrt(bias2+varN)



#---------------------------------------------------------------------
#
# Generates numerical pde related stuff
#
#---------------------------------------------------------------------
n=64;       #number of grid points on each component
h=1/n;
e = np.ones(n,);
data = np.array([-e,2*e,-e])
diags = np.array([-1, 0, 1])
k=spdiags(data, diags, n, n).toarray()
I=np.eye(n);
A=(scipy.sparse.kron(k,I)+scipy.sparse.kron(I,k))/(h**2);
A=A.tocsr()


# Redefines the forcing term
def F(X):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1,n)
    a=0.5*(np.exp(-1000*(x[:,None]-X[0])**2-1000*(y[None,:]-X[1])**2))
    return a.flatten()



def Obs(u):
     return u;
    


sigma=np.load('noise_elliptic.npy')
d=np.load('data_elliptic.npy')
mx=np.load('true_mean_x_elliptic.npy')
my=np.load('true_mean_y_elliptic.npy')
true_th=np.array([mx,my])
#---------------------------------------------------------------------
#
# Defines posterior
#---------------------------------------------------------------------
#lower and upper limit for uniform prior
a=np.array([0,0]);
b=np.array([1,1]);
#Defines likelihood, prior and posterior functions
def L(x):
    u = spsolve(A,F(x))
    return -0.5*sum(abs(d-Obs(u))**2)*(h**2)/(sigma**2.0);
def pr(x):
    return np.log(float(np.all(x>a))*float(np.all(x<b)));
def post(x):
    return L(x)+pr(x)

#---------------------------------------------------------------------
#
# Defines some hyper parameters, Number of temperatures, N samples, etc.
#
#---------------------------------------------------------------------
p=2;#number of parameters
N=10000#int(10**4); #number of samples
N_temp=4;#number of temperatures
Ns=1; #How often do we swap
X=np.zeros([N,p,N_temp])#prealloactes
y=np.zeros([N_temp,p]); #preallocates proposals
bt=np.zeros([N,N_temp])
T0=400.0**(1.0/3.0)
beta = np.array(T0**np.arange(0,N_temp));
beta_original = np.copy(beta);
x0=np.random.random((p,N_temp))
sigma_is=np.array([[0.035,0.035],[0.15,0.15], [0.4,0.4],[0.6,0.6]]);#*linspace(1,100,N_temp);





mean_r=np.zeros((Nruns,2))
mean_pf=np.zeros((Nruns,2))
mean_uw=np.zeros((Nruns,2))
mean_z=np.zeros((Nruns,2))



mx=np.load('true_mean_x_elliptic.npy')
my=np.load('true_mean_y_elliptic.npy')

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
    Xptf=base.full_vanilla(post,N,beta,sigma_is,x0,Ns=1,Disp=0) #full vanilla, reversible
    Xuw=base.unweighted_IS(post,N,beta,sigma_is,x0,1,1,Disp=0) #unweighted Is
    Xw,W_IS,_=base.weighted_IS(post,N,beta_original,sigma_is,x0,Disp=0) #weighted IS
    Xrwm=base.rwm(post,N*N_temp,sigma_is[1],x0[:,0],Disp=0) #random walk metropolis
    yy,ww=base.weight_samples(Xw,W_IS)
    xx=base.resample_IS(yy,ww,N)
    
    mean_r[i,:]=np.mean(Xrwm[N_temp*Burn_in:],0)
    mean_pf[i,:]=np.mean(Xptf[Burn_in:,:,0],0)

    mean_uw[i,:]=np.mean(Xuw[Burn_in:,:,0],0)
    mean_z[i,:]=base.mean(Xw,W_IS)
    
    np.save('mean_r_elliptic_b.npy',mean_r)
    np.save('mean_p_elliptic_b.npy',mean_pf)
    np.save('mean_uw_elliptic_b.npy',mean_uw)  
    np.save('mean_z_elliptic_b.npy',mean_z)    
    
print('----------------------------------')
print('----------------------------------')
print('means')
print(np.mean(mean_r,0))
print(np.mean(mean_pf,0))
print(' GPT ')

print(np.mean(mean_uw,0))
print(np.mean(mean_z,0))


print('----------------------------------')
print('variances')
print(np.var(mean_r,0))
print(np.var(mean_pf,0))
print(' GPT ')

print(np.var(mean_uw,0))
print(np.var(mean_z,0))


print('----------------------------------')
print('bias ^2 ')
print(np.abs((np.mean(mean_r,0)-true_th))**2.0)
print(np.abs((np.mean(mean_pf,0)-true_th))**2.0)
print(' GPT ')

print(np.abs((np.mean(mean_uw,0)-true_th))**2.0)
print((np.abs(np.mean(mean_z,0)-true_th))**2.0)

print('----------------------------------')
print('MSE ')
print(np.var(mean_r,0)+(np.mean(mean_r,0)-true_th)**2.0)
print(np.var(mean_pf,0)+(np.mean(mean_pf,0)-true_th)**2.0)
print(' GPT ')
print(np.var(mean_uw,0)+(np.mean(mean_uw,0)-true_th)**2.0)
print(np.var(mean_z,0)+(np.mean(mean_z,0)-true_th)**2.0)




