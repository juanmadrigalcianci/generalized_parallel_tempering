

import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../") )
# from hippylib import *

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)


def MatMatMult(A,B):
    """
    Compute the matrix-matrix product :math:`AB`.
    """
    Amat = dl.as_backend_type(A).mat()
    Bmat = dl.as_backend_type(B).mat()
    out = Amat.matMult(Bmat)
    rmap, _ = Amat.getLGMap()
    _, cmap = Bmat.getLGMap()
    out.setLGMap(rmap, cmap)
    return dl.Matrix(dl.PETScMatrix(out))


class prior_measure:
    
    def __init__(self,mesh,delta=0.5,gamma=0.1,mean=0,scale=1):
        self.mean=mean
        self.scale=scale
        V = dl.FunctionSpace(mesh, 'Lagrange', 1)
        self.V=V
        alpha=math.pi/4.0
        theta0=2.
        theta1=0.5
        def bilap(trial, test,Theta,delta,gamma): 
                if Theta == None:
                    varfL = ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx
                else:
                    varfL = ufl.inner( Theta*ufl.grad(trial), ufl.grad(test))*ufl.dx       
                varfM = ufl.inner(trial,test)*ufl.dx
                varf_robin = ufl.inner(trial,test)*ufl.ds
                robin_coeff = gamma*np.sqrt(delta/gamma)/1.42
        
                
                return dl.Constant(gamma)*varfL + dl.Constant(delta)*varfM + dl.Constant(robin_coeff)*varf_robin

        AN=np.zeros((2,2))
        AN[0,0]=theta0*np.sin(alpha)**2.0
        AN[0,1]=(theta0-theta1)*np.sin(alpha)*np.cos(alpha)
        AN[1,0]=(theta0-theta1)*np.sin(alpha)*np.cos(alpha)
        AN[1,1]=theta1*np.cos(alpha)**2.0

        anis_diff = dl.Expression((('t0*sa*sa+t1*ca*ca','(t0-t1)*sa*ca'),
                              ('(t0-t1)*sa*ca','t0*ca*ca+t1*sa*sa')), t1=theta1,t0=theta0,sa=np.sin(alpha),ca=np.cos(alpha),degree=1)
        
        # anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
        
        # import pdb
        # pdb.set_trace()
        
        # anis_diff.set(theta0, theta1, alpha)
        
        v=dl.TestFunction(V)
        u=dl.TrialFunction(V)
        test=dl.TestFunction(V)
        
        varfK=bilap(u,v,anis_diff,delta,gamma)
        self.A=dl.assemble(varfK)
        
        
        Qh = dl.FunctionSpace(mesh, 'Lagrange',1)
                    
        ph = dl.TrialFunction(Qh)
        qh = dl.TestFunction(Qh)
        Mqh = dl.assemble(ufl.inner(ph,qh)*ufl.dx)
        one_constant = dl.Constant(1.)
        ones = dl.interpolate(one_constant, Qh).vector()
        dMqh = Mqh*ones
        Mqh.zero()
        dMqh.set_local( ones.get_local() / np.sqrt(dMqh.get_local() ) )
        Mqh.set_diagonal(dMqh)
        MixedM = dl.assemble(ufl.inner(ph,test)*ufl.dx)
        self.sqrtM = MatMatMult(MixedM, Mqh)
        #np.random.seed(1)

        

    def sample(self,exp=False):
        
        x=dl.Vector()
        samp=dl.Vector()
        
        self.sqrtM.init_vector(x, 1)
        self.sqrtM.init_vector(samp, 1)
        
        dim=len(x)
        x[:]=np.random.standard_normal(dim)
        rhs=self.sqrtM*x
        dl.solve(self.A,samp,rhs)
        if exp==True:
            samp=np.exp(samp)
        return self.mean +self.scale*samp
        
    def plot_prior_sample(self,exp=False):
            samp=self.sample(exp)
            s=dl.Function(self.V)
            s.vector()[:]=samp[:]
            if exp==False:
                p=dl.plot(s)
                plt.colorbar(p)
            else:
                p=dl.plot(dl.exp(s))
                plt.colorbar(p)
    
    def plot_prior(self,samp,exp=False,cmap='Spectral'):
        s=dl.Function(self.V)
        s.vector()[:]=samp[:]
        if exp==False:
            p=dl.plot(s,cmap=cmap)
            plt.colorbar(p)
        else:
            p=dl.plot(dl.exp(s))
            plt.colorbar(p)
        return p
