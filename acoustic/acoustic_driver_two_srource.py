#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:51:30 2020

@author: juan
"""

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
import ufl
import time
from scipy import interpolate


def acoustic_main(theta,plot=False,level=0):
    
    
    
    """
    
    
    theta is the parameter vector. here we have 
    
    
    theta 0 = x1
    theta 1 = x2
    theta 2 = alpha
    theta 3 = beta
    
    
    """
    
    
    x1=theta[0]
    x2=theta[1]
    alpha=theta[2]**2.0
    beta=theta[3]**2.0
    
    ML=2**level
    #c=500
    Nx=40*ML
    Ny=40*ML
    p0=dl.Point(0.,0.)
    p1=dl.Point(3,2)
    
    rx=[1.,1.5,2.]
    ry=[1.99,1.99,1.99]
    
    Nrx=len(rx)
    Nry=len(ry)
    Nr=Nry
    t0=time.time()
    #beta/alpha=c^2=500**2
    #beta=5000**2
    #alpha=10**2
    
    
    #defines the source model 
    def source(t,x1,x2):
        delta =dl.Expression('M*exp(-(x[0]-x1)*(x[0]-x1)/(a*a)-(x[1]-x2)*(x[1]-x2)/(a*a))/a*(1-2*pi2*f02*t*t)*exp(-pi2*f02*t*t)'
                         ,pi2=np.pi**2,a=1E-1, f02=f02, M=1E10,x1=x1,x2=x2,t=t,degree=1)
        return delta
    
    
    
    
    
    
    
    B=dl.Constant(beta)
    A=dl.Constant(alpha)
    
    
    mesh = dl.RectangleMesh(p0,p1,Nx,Ny)
    V=dl.FunctionSpace(mesh, "Lagrange", 1)
    c2=beta/alpha
    hmin=mesh.hmin()
    
    
    dt=0.15*hmin/(c2**0.5)
    
    # Time variables
    
    t = 0; T = 0.003
    Nt=int(np.ceil(T/dt))
    if plot:
        print('value of Nt is '+str(Nt))
        print('dt is '+str(dt))

    time_=np.zeros(Nt)

    U_wave=np.zeros((Nt,Nr))
    # Previous and current solution
    u1= dl.interpolate(dl.Constant(0.0), V)
    u0= dl.interpolate(dl.Constant(0.0), V)
    
    # Variational problem at each time
    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)
    M, K = dl.PETScMatrix(), dl.PETScMatrix()# Assembles matrices
    M=dl.assemble(A*u*v*dl.dx,tensor=M)
    f02=1000**2.0
    K=dl.assemble(dl.inner(B*dl.grad(u),dl.grad(v))*dl.dx,tensor=K)
    
    # M=dl.assemble(u*v*dl.dx)
    # K=dl.assemble(dl.inner(dl.grad(u),dl.grad(v))*dl.dx)
    delta =source(t,x1,x2)
    

    
    f=dl.interpolate(delta,V)
    # ABC
    class ABCdom(dl.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[1] < 2.0)
    
    abc_boundaryparts = dl.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    ABCdom().mark(abc_boundaryparts, 1)
    #self.ds = Measure("ds")[abc_boundaryparts]
    ds = dl.Measure('ds', domain=mesh, subdomain_data=abc_boundaryparts)
    weak_d = dl.inner((A*B)**0.5*u,v)*ds(1)
    class_bc_abc = ABCdom()    # to make copies
    # builds the ABS matrix 
    D = dl.assemble(weak_d)
    
    #saves
    if plot:
        
        ofile=dl.File('output/ud_.pvd')
    
    u=dl.Function(V)
    ti=0
    while t <= T:
        fv=dl.assemble(f*v*dl.dx)
        Kun=K*u1.vector()
        Dun=D*(u1.vector()-u0.vector())/dt
        b=fv-Kun-Dun
        dl.solve(M, u.vector(), b)
        # dl.plot(u);plt.show()
        # import pdb
        # pdb.set_trace()
        u.vector()[:]=dt**2.0*u.vector()[:]+2.0*u1.vector()[:]-u0.vector()[:]
        #u=dt**2*u+2.0*u1-u0
        u0.assign(u1)
        u1.assign(u)
        
        for rec in range(Nr):
            U_wave[ti,rec]=u([rx[rec],ry[rec]])
        time_[ti]=t
        t += dt
        ti+=1
        
        delta =source(t,x1,x2)

    
        f=dl.interpolate(delta,V)
        # Reduce the range of the solution so that we can see the waves
        if plot:
            ofile << (u, t)
            
    #print('Total time '+str(round(time.time()-t0,3)))
    
    
    return U_wave,time_




def wave_int(theta,level=0,plot=False):
    uwave,time_=acoustic_main(theta,level=level,plot=plot)
    Ncol=uwave.shape[1]
    
    f=[]
    for i in range(Ncol):
        ff=interpolate.interp1d(time_,uwave[:,i],fill_value="extrapolate")
        f.append(ff)
    return f,uwave,time_

def trapezoidal(y,dt):
    I=y[0]+y[-1]+np.sum(2*y[1:-1])
    return I*0.5*dt
    
    
    
    

def compute_misfit(theta,y,times):
    Nr=y.shape[1]
    Nt=y.shape[0]
    dt=times[1]-times[0]
    dt=1000*dt
    
    f,_,_=wave_int(theta)
    urec=np.zeros((Nt,Nr))
    # interpolates to the proper times
    for i in range(Nr):
        urec[:,i]=f[i](times)
    #computes the missfit
    misfit=0
    for i in range(Nr):
        misfit+=trapezoidal((y[:,i]-urec[:,i])**2.0,dt)
    misfit=misfit/Nr
    return misfit


def prior(theta,a,b):
    return np.log(float(np.all(theta>a))*float(np.all(theta<b)));


def posterior(theta,y,times,a,b,sigma):
    L=compute_misfit(theta, y, times)*0.5/(sigma**2.0)
    Pr=prior(theta,a,b)
    return -L+Pr
    
    
    
    
##### testes the codes
    

# defines parameters

#a=np.array([0.,0.,8, 4800] )
#b=np.array([3.,2.,12,5200])
#p=len(a)






#generates data
#th_true=a+(b-a)*np.random.random(p)
#y_true,tt=acoustic_main(th_true)
#nt=len(tt);nr=y_true.shape[1]
#noise_ratio=0.02
#tt[-1]=0.004
#sigma=noise_ratio*np.max(np.abs(y_true))
#y_data=y_true+sigma*np.random.standard_normal((nt,nr))
#plt.plot(tt,y_data)

#t0=time.time()
# samples randomly from the prior
#theta_test=a+(b-a)*np.random.random(p)
#post=posterior( theta_test, y_data, tt, a, b,sigma)


#tf=time.time()-t0
#print('post is ' +str(post))
#print('time '+str(tf))
#print('random theta is '+str(theta_test))
#print('true   theta is '+str(th_true))

#ur,tr=acoustic_main(theta_test,plot=True)
#plt.show()
#plots 
#for i in range(nr):
#    plt.plot(tt,y_data[:,i])
#    plt.plot(tr,ur[:,i])
#    plt.show()

    
    
    
    
    







    
    
    
