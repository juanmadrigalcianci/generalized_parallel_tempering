#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:56:29 2021

@author: juan
"""

import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import prior2 as prior
import logging
import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve




t0=time.time()
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)


#%%

def forward(theta,l=0,plot=False):
    # discretization
    Nx=int(28*2.**l)
    Ny=int(28*2.**l)
    
    #domain
    p0=dl.Point(0.,0.)
    p1=dl.Point(3,2)
    # creates mesh and function space
    mesh = dl.RectangleMesh(p0,p1,Nx,Ny)
    V=dl.FunctionSpace(mesh, "Lagrange", 1)
    
    
    #receiver location
    N_rec_x=5
    rx=np.linspace(1,2,N_rec_x)
    ry=2*np.ones(len(rx))
    
    # Source location
    x1=1.5
    x2=1.
    
    cm=4
    c_mult=0.1*cm
    #Obtains C2
    
    #PRIOR=prior.prior_measure(mesh)
    
    
    #C2=PRIOR.sample(exp=True)
    #C2=cm+c_mult*np.exp(theta)
    
    C2=(theta)
    c=dl.Function(V)
    c.vector()[:]=C2[:]
    #dl.plot(c)
    
    A=dl.Constant(1.0)
    
    #PRIOR.plot_prior(C2)
    
    
    
    hmin=mesh.hmin()
    
    
    dt=0.1*hmin/(cm**2.0)
    #%%
    B=10.+c*c
    
    
    Nrx=len(rx)
    Nry=len(ry)
    Nr=Nry
    
    #beta/alpha=c^2=500**2
    #beta=5000**2
    #alpha=10**2
        
    #defines the source model 
    def source(t,x1,x2):
        delta =dl.Expression('M*exp(-(x[0]-x1)*(x[0]-x1)/(a*a)-(x[1]-x2)*(x[1]-x2)/(a*a))/a*(1-2*pi2*f02*t*t)*exp(-pi2*f02*t*t)'
                         ,pi2=np.pi**2,a=6E-2, f02=f02, M=1E5,x1=x1,x2=x2,t=t,degree=1)
        return delta
        
    
    
    # Time variables
    
    t = 0; T =0.6
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
    
    # mass_form = A*u*v*dl.dx
    # mass_action_form = dl.action(mass_form, dl.Constant(1))
    
    # M_consistent = dl.assemble(mass_form)
    # print("Consistent mass matrix:\n", np.array_str(M_consistent.array(), precision=3))
    
    # M_lumped = dl.assemble(mass_form)
    # M_lumped.zero()
    # M_lumped.set_diagonal(dl.assemble(mass_action_form))
    # print("Lumped mass matrix:\n", np.array_str(M_lumped.array(), precision=3))
    # M=M_lumped
    
    if plot:
        
        ofile=dl.File('output/ud_.pvd')
    
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
    #class_bc_abc = ABCdom()    # to make copies
    # builds the ABS matrix 
    D = dl.assemble(weak_d)
    
    # Find inverse
    mm=sp.csc_matrix(M.array())
    I=sp.csc_matrix(np.eye(len(M.array())))
    Minv=spsolve(mm, I)
    
    #%%
    
    qoi=0
    
    #saves
    if plot:
        
        ofile=dl.File('output/ud_.pvd')
    
    u=dl.Function(V)
    ti=0
    while t <= T:
        # if ti%100==0:
        #     print('time step '+str(ti)+' out of '+str(Nt))
        fv=dl.assemble(f*v*dl.dx)
        Kun=K*u1.vector()
        Dun=D*(u1.vector()-u0.vector())/dt
        b=fv-Kun-Dun
        #B=dl.Function(V)
        #B.vector()[:]=b[
        u.vector()[:]=Minv@b[:]
        #dl.solve(M, u.vector(), b)
        
        # M_vect = dl.assemble(mass_action_form)
        # u = dl.Function(V)
        # u.vector().set_local(B.vector().get_local()/M_vect.get_local())    
        
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
        
        f =source(t,x1,x2)
        
    
        #f=dl.interpolate(delta,V)
        # Reduce the range of the solution so that we can see the waves
        if plot:
            ofile << (u, t)
    qoi1=dl.exp(dl.assemble(c*dl.dx))
    qoi2=dl.assemble(dl.exp(c)*dl.dx)
    qoi3=dl.assemble(B*dl.dx)
    qoi4=6.0+dl.assemble(c*dl.dx)
    qoi5=np.max(c.vector()[:])
    qoi6=10.0-c([1.5,1])
    qoi7=np.exp(c([1.5,1]))
    
    
    # import pdb
    # pdb.set_trace()
    
    qoi=np.array((qoi1,qoi2,qoi3,qoi4,qoi5,qoi6,qoi7))
    
    
    
    
    #np.save('wave.npy',U_wave)
    return U_wave,    qoi
#print('Total time '+str(round(time.time()-t0,3)))
    
    
