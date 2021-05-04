#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:04:16 2021

@author: juan
"""
import numpy as np
import base as base
import time
import math
import wave_generate_data as ww
import matplotlib.pyplot as plt
import dolfin as dl
from matplotlib import rc
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
Nx=28
Ny=28
p0=dl.Point(0.,0.)
p1=dl.Point(3,2)
# creates mesh and function space
mesh = dl.RectangleMesh(p0,p1,Nx,Ny)
V=dl.FunctionSpace(mesh, "Lagrange", 1)
field=ww.th_true()
v2f=dl.vertex_to_dof_map(V)
X,Y=np.meshgrid(np.linspace(0,3,29),np.linspace(0,2,29))

N_rec_x=5
rx=np.linspace(1,2,N_rec_x)
ry=1.95*np.ones(len(rx))
plt.figure(figsize=(9,4))
S = plt.contour(X,Y,field[v2f].reshape((29,29)),levels = [0],
                 colors=('k',),linestyles=('-',),linewidths=(2,))
vmin=-3.5
vmax=3.5
p=plt.contourf(X,Y,field[v2f].reshape((29,29)),levels = np.linspace(vmin, vmax, 50+1),cmap='Spectral')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.colorbar(p)

for i in range(N_rec_x):
    for j in range(N_rec_x):
        plt.plot(rx[i],ry[j],'*',color='m')

plt.savefig('true_field.png')
