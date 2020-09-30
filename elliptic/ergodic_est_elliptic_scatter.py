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
import math
#------------------------   parallel libraries     -------------------
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#name=MPI.Get_processor_name()
#---------------------------------------------------------------------


# Number of runs 
Nruns=1


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
N=25000#int(10**4); #number of samples
N_temp=4;#number of temperatures
Ns=1; #How often do we swap
X=np.zeros([N,p,N_temp])#prealloactes
y=np.zeros([N_temp,p]); #preallocates proposals
bt=np.zeros([N,N_temp])
T0=400.0**(1.0/3.0)
beta = np.array(T0**np.arange(0,N_temp));
beta_original = np.copy(beta);
x0=np.random.random((p,N_temp))
sigma_is=np.array([[0.03,0.03],[0.1,0.1], [0.4,0.4],[0.6,0.6]]);#*linspace(1,100,N_temp);
#sigma_is=np.array([[0.1,0.1],[0.15,0.15], [0.3,0.3],[0.45,0.45]])#*linspace(1,100,N_temp);




mean_r=np.zeros((Nruns,2))
mean_p=np.zeros((Nruns,2))
mean_uw=np.zeros((Nruns,2))
mean_w=np.zeros((Nruns,2))
mean_pf=np.zeros((Nruns,2))
mean_sd=np.zeros((Nruns,2))

mean_y=np.zeros((Nruns,2))
mean_yj=np.zeros((Nruns,2))

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
    
    Xsd=base.state_dependent_PT(post,N,beta,sigma_is,x0,Ns=1,Disp=0) #full vanilla, reversible

    
    Xptf=base.full_vanilla(post,N,beta,sigma_is,x0,Ns=1,Disp=0) #full vanilla, reversible
    Xuw=base.unweighted_IS(post,N,beta,sigma_is,x0,1,1,Disp=0) #unweighted Is
    Xw,W_IS,W_IS2=base.weighted_IS(post,N,beta_original,sigma_is,x0,Disp=0) #weighted IS
    Xrwm=base.rwm(post,N*N_temp,0.16,x0[:,0],Disp=0) #random walk metropolis
    yy,ww=base.weight_samples(Xw,W_IS)
    xx=base.resample_IS(yy,ww,N)
    
    mean_r[i,:]=np.mean(Xrwm[N_temp*Burn_in:],0)
    mean_pf[i,:]=np.mean(Xptf[Burn_in:,:,0],0)
    mean_sd[i,:]=np.mean(Xsd[Burn_in:,:,0],0)

    mean_uw[i,:]=np.mean(Xuw[Burn_in:,:,0],0)
    mean_w[i,:]=np.average(yy[Burn_in*math.factorial(N_temp):,:,0],0,weights=ww[Burn_in*math.factorial(N_temp):])

    mean_y[i,:]=np.mean(xx[Burn_in:,:,0],0)
    mean_z[i,:]=base.mean(Xw[Burn_in:,:,:],W_IS[Burn_in:,:])
    
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
print(np.mean(mean_z,0))


print('----------------------------------')
print('variances')
print(np.var(mean_r,0))
print(np.var(mean_pf,0))
print(np.var(mean_sd,0))

print(' GPT ')

print(np.var(mean_uw,0))
print(np.var(mean_w,0))
print(np.var(mean_y,0))
print(np.var(mean_z,0))


print('----------------------------------')
print('bias ^2 ')
print(np.abs((np.mean(mean_r,0)-true_th))**2.0)
print(np.abs((np.mean(mean_pf,0)-true_th))**2.0)
print(np.abs((np.mean(mean_sd,0)-true_th))**2.0)

print(' GPT ')

print(np.abs((np.mean(mean_uw,0)-true_th))**2.0)
print((np.abs(np.mean(mean_w,0)-true_th))**2.0)
print((np.abs(np.mean(mean_y,0)-true_th))**2.0)
print((np.abs(np.mean(mean_z,0)-true_th))**2.0)

print('----------------------------------')
print('MSE ')
print(np.var(mean_r,0)+(np.mean(mean_r,0)-true_th)**2.0)
print(np.var(mean_pf,0)+(np.mean(mean_pf,0)-true_th)**2.0)
print(np.var(mean_sd,0)+(np.mean(mean_sd,0)-true_th)**2.0)

print(' GPT ')
print(np.var(mean_uw,0)+(np.mean(mean_uw,0)-true_th)**2.0)
print(np.var(mean_w,0)+(np.mean(mean_w,0)-true_th)**2.0)
print(np.var(mean_y,0)+(np.mean(mean_y,0)-true_th)**2.0)
print(np.var(mean_z,0)+(np.mean(mean_z,0)-true_th)**2.0)

import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interpn
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

fig, axes = plt.subplots(nrows = 1, ncols = 5, figsize = (12,4))
plt.suptitle('Scatterplots')

axes[0]=density_scatter( Xrwm[BI:4*Max,0], Xrwm[BI:4*Max,1], ax=axes[0],bins = [100,100], cmap='Spectral')
axes[0].set_title(r'RWM')
axes[0].set_xlim(0,1)
axes[0].set_ylim(0,1)


axes[1]=density_scatter( Xptf[BI:Max,0,0], Xptf[BI:Max,1,0], ax=axes[1],bins = [100,100], cmap='Spectral')
axes[1].set_title(r'PT')
axes[1].set_xlim(0,1)
axes[1].set_ylim(0,1)

axes[2]=density_scatter( Xuw[BI:Max,0,0], Xuw[BI:Max,1,0], ax=axes[2],bins = [100,100], cmap='Spectral')
axes[2].set_title(r'UGPT')
axes[2].set_xlim(0,1)
axes[2].set_ylim(0,1)

axes[3]=density_scatter( xx[BI:Max,0,0], xx[BI:Max,1,0] ,ax=axes[3],bins = [100,100], cmap='Spectral')
axes[3].set_title(r'WGPT')
axes[3].set_xlim(0,1)
axes[3].set_ylim(0,1)

axes[4]=density_scatter( Xw[BI:Max,0,0], Xw[BI:Max,1,0],ax=axes[4],bins = [100,100], cmap='Spectral')
axes[4].set_title(r'WGPT (inv)')
axes[4].set_xlim(0,1)
axes[4].set_ylim(0,1)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


