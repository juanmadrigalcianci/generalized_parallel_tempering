#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:07:56 2019

@author: jmadriga
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import scipy





t=0
T=[  1.        ,   7.36  ,  54.28, 400.        ]
fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (12,2.7))
for ax in axes:
    T[t]=round(T[t],2)
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
    
    def Obs(u):
        return(u)
    # Redefines the forcing term
    def F(X):
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1,n)
        a=0.5*(np.exp(-1000*(x[:,None]-X[0])**2-1000*(y[None,:]-X[1])**2))
        return a.flatten()
        
    
        
    #for simplicity, let's use the whole solution as an observation operator
    sigma=np.load('noise_elliptic.npy')
    d=np.load('data_elliptic.npy')
    
    #lower and upper limit for uniform prior
    a=np.array([0,0]);
    b=np.array([1,1]);
    #Defines likelihood, prior and posterior functions
    def L(x):
        u = spsolve(A,F(x))
        return -0.5*sum(abs(d-Obs(u))**2)*(h**2)/(sigma**2);
    def pr(x):
        return np.log(float(np.all(x>a))*float(np.all(x<b)));
    def post(x):
        return L(x)+pr(x)
    
    def normal(x,y,T=1):
        return  np.exp(post([x,y])/T)
    if t==0:
        N=200
    else:
        N=100
    am=0
    bm=1
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
    ax.set_title(r'$\mu_{'+str(t+1)+'}$, $T_'+str(t+1)+'='+str(T[t])+'$')
    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$\\theta_2$')
    plt.tight_layout(True)
    fig.colorbar(p, ax=ax)
    #ax.set_colorbar(p)
    t=t+1
plt.savefig('true_elliptic_density_'+str(t)+'.png', format='png',bbox_inches='tight', transparent=True)
plt.show()
            

