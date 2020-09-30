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

def normal(x,y):
    return  np.exp(post([x,y]))

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
        z[i,j]=normal(x[i],x[j])
        my+=x[j]*z[i,j]
        mx+=x[i]*z[i,j]
        
Z=np.sum(z*dh**2)                        
Mx=(mx*dh**2)/Z
My=(my*dh**2)/Z
X,Y=np.meshgrid(x,x)
p=plt.contourf(X,Y,z.T/Z,100, cmap='Spectral')
plt.title('Density quarter circle')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')

plt.colorbar(p)
plt.savefig('true_cirlce_density_t.pdf', format='pdf',bbox_inches='tight', transparent=True)
np.save('true_mean_x_Circle.npy',Mx)
np.save('true_mean_y_Circle.npy',My)

