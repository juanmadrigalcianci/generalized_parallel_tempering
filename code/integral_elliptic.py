#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the true expected values for the elliptic problem 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import scipy
import time as time
import matplotlib.pyplot as plt
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

def F(X):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1,n)
    a=0.25*(np.exp(-1000*(x[:,None]-X[0])**2-1000*(y[None,:]-X[1])**2)+
            np.exp(-1000*(x[:,None]-X[2])**2-1000*(y[None,:]-X[3])**2)+
            np.exp(-1000*(x[:,None]-X[4])**2-1000*(y[None,:]-X[5])**2)+
            np.exp(-1000*(x[:,None]-X[6])**2-1000*(y[None,:]-X[7])**2))
    return a.flatten()


#---------------------------------------------------------------------
#
# Generates Data
#
#---------------------------------------------------------------------
th_true=[.2,.2,.2,.8,.8,.2,.8,.8];
f=F(th_true);
t0=time.time()
u_true=spsolve(A,f);
# Test the solution
print(time.time()-t0)

# Redefines the forcing term
def F(X):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1,n)
    a=0.5*(np.exp(-1000*(x[:,None]-X[0])**2-1000*(y[None,:]-X[1])**2))
    return a.flatten()

#for simplicity, let's use the whole solution as an observation operator
sigma=0.01*max(np.abs(u_true));

d=u_true+sigma*np.random.standard_normal(np.size(u_true),); #adds noise to the solution


# Saves the true noise and all
np.save('data_elliptic.npy',d)
np.save('utrue_elliptic.npy',u_true)
np.save('noise_elliptic.npy',sigma)



X,Y=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
p=plt.contourf(X,Y,np.reshape(d,[n,n]),100, cmap='Spectral')
plt.title(r'$y$ elliptic')
plt.colorbar(p)
plt.savefig('measured_data_density.pdf', format='pdf',bbox_inches='tight', transparent=True)




def Obs(u):
    return(u)
# Redefines the forcing term
def F(X):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1,n)
    a=0.5*(np.exp(-1000*(x[:,None]-X[0])**2-1000*(y[None,:]-X[1])**2))
    return a.flatten()



#for simplicity, let's use the whole solution as an observation operator
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

def normal(x,y):
    return  np.exp(post([x,y]))

N=200
am=0
bm=1
x=np.linspace(0,1,N)
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
plt.title('Density elliptic')
plt.colorbar(p)
plt.savefig('true_elliptic_density.pdf', format='pdf',bbox_inches='tight', transparent=True)
np.save('true_mean_x_elliptic.npy',Mx)
np.save('true_mean_y_elliptic.npy',My)










