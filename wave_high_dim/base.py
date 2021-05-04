#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:21:19 2019

@author: jmadriga 2019

"""





import numpy as np
import itertools as it
from scipy.special import logsumexp
from pdb import set_trace
import time as time
import math
"""

Creates the base for RWM

takes as input

 post: funciton evaluated at x
 N:    Number of samples
 beta: array of temperatures
 sigma_rwm: matrix of covariance
 x0:   initial value of length (N_temp,p)
 Ns: How often to swap
 Disp optimal: displays progress


outputs:

X array of (N,P,N_temp) samples


"""

def display(x,a,i,post,mod=1):
    
    if np.mod(i,mod)==0:
    
        print('--------------')
        print('iteration '+str(i))
        print('current ' +str(x))
        print('post ' +str(post))
        print('ar ' +str(a/i))
    return 
    
    
    


def rwm(post,N,sigma_rwm,x0,Disp=0):

    #preallocation:

    p=np.size(x0,0)

    #prealocates posteriors
    px=np.zeros(N)
    # preallocates number of samnples
    X=np.zeros((N,p))



    X[0,:]=x0

    acpt=0
    #gets first posterior
    px[0]=post(X[0,:])

    #starts the main MCMC loop

    for j in range(N-1):

    #computes posterior
        y=X[j,:]+sigma_rwm*np.random.standard_normal(p,)
        py=post(y)

    #accepts-rejects

        ratio=py-px[j];
        ####set_trace()
        if (np.log(np.random.random(1))<ratio):
            X[j+1,:]=y
            px[j+1]=py
            acpt+=1
        else:
            X[j+1,:]=X[j,:]
            px[j+1]=px[j]
            
        if Disp:
            display(X[j+1,:], acpt, j+1,px[j+1])
            
            
            
            
            

    print(acpt/N)
    return X




"""

Creates the base for pCN

"""

def pcn(post,N,rho,PRIOR,x0,Disp=0,dim_q=1):

    #preallocation:

    p=np.size(x0,0)

    #prealocates posteriors
    px=np.zeros(N)
    # preallocates number of samnples
    X=np.zeros((N,p))
    qoi=np.zeros((N,dim_q))
    rho_shift=(1-rho**2.0)**0.5


    X[0,:]=x0

    acpt=0
    #gets first posterior
    px[0],qoi[0]=post(X[0,:])

    #starts the main MCMC loop

    for j in range(N-1):

    #computes posterior
        y=rho_shift*X[j,:]+rho*PRIOR.sample(exp=False)
        py,qoi_y=post(y)

    #accepts-rejects

        ratio=py-px[j];
        ####set_trace()
        if (np.log(np.random.random(1))<ratio):
            X[j+1,:]=y
            px[j+1]=py
            qoi[j+1,:]=qoi_y
            acpt+=1
        else:
            X[j+1,:]=X[j,:]
            px[j+1]=px[j]
            qoi[j+1,:]=qoi[j]
            
        if Disp:
            display('pCN',acpt, j+1,px[j+1])
            
            
            
            
            

    print(acpt/N)
    return X,qoi



"""

Creates the base for parallel tempering based on RWM

takes as input

 post: funciton evaluated at x
 N:    Number of samples
 beta: array of temperatures
 sigma_rwm: matrix of covariance
 x0:   initial value of length (N_temp,p)
 Ns: How often to swap
 Disp optimal: displays progress


outputs:

X array of (N,P,N_temp) samples


"""
                        
def swap_strategy_pcn(X,px,qoi,beta,strategy,j):
    N_temp=len(beta)
    if strategy=='full_vanilla_forward':
        #swaps from top to bottom 
        for i in range(N_temp-1):
            #computes ak
            ak=px[j,i]/beta[i+1]+px[j,i+1]/beta[i]-px[j,i]/beta[i]-px[j,i+1]/beta[i+1]

            if ak>np.log(np.random.random(1)):
            #changes posteriors
                p1=np.copy(px[j,i])
                p2=np.copy(px[j,i+1])
                q1=np.copy(qoi[j,:,i])
                q2=np.copy(qoi[j,:,i+1])
                
                #Changes posts
                px[j,i]=p2
                px[j,i+1]=p1
                
                #changes QoI
                qoi[j,:,i]=q2
                qoi[j,:,i+1]=q1                
                #changes positions
                p1=np.copy(X[j,:,i])
                p2=np.copy(X[j,:,i+1])                
                X[j,:,i]=p2; 
                X[j,:,i+1]=p1; 
    

    elif strategy=='full_vanilla_backward':
        #print('------')
        #swaps from bottom to top
        ###set_trace()
        for i in range(N_temp-1,0,-1):
            #computes ak
            ak=px[j,i]/beta[i-1]+px[j,i-1]/beta[i]-px[j,i]/beta[i]-px[j,i-1]/beta[i-1]
            if ak>np.log(np.random.random(1)):
                #print('swapped '+str(i)+' and '+str(i-1))
                
                #if i==1: print(X[j,:,i-1]);print(X[j,:,i])
                
                #changes posteriors
                p1=np.copy(px[j,i])
                p2=np.copy(px[j,i-1])
                
                #copies qois
                q1=np.copy(qoi[j,:,i])
                q2=np.copy(qoi[j,:,i-1])
                
                
                
                px[j,i]=p2
                px[j,i-1]=p1
                
                
                #changes qoi
                qoi[j,:,i]=q2
                qoi[j,:,i-1]=q1                
                
                
                #changes positions
                p1=np.copy(X[j,:,i])
                p2=np.copy(X[j,:,i-1])                
                X[j,:,i]=p2; 
                X[j,:,i-1]=p1;  
                
                
            
    return X,px,qoi
    

def swap_strategy(X,px,beta,strategy,j):
    N_temp=len(beta)
    if strategy=='vanilla':
        #Pics one random index
        i=np.random.randint(N_temp-1)
        #computes ak
        ak=px[j,i]/beta[i+1]+px[j,i+1]/beta[i]-px[j,i]/beta[i]-px[j,i+1]/beta[i+1]
        if ak>np.log(np.random.random(1)):
        #changes posteriors
            p1=np.copy(px[j,i])
            p2=np.copy(px[j,i+1])
            px[j,i]=p2
            px[j,i+1]=p1
            
            #changes positions
            p1=np.copy(X[j,:,i])
            p2=np.copy(X[j,:,i+1])                
            X[j,:,i]=p2; 
            X[j,:,i+1]=p1; 
    elif strategy=='vanilla_random':
     #swaps t0 with ti chosen at random
     #samples a random temperature other than t0 and attempts to swap
        i=np.random.randint(1,N_temp)
        ak=px[j,i]/beta[0] + px[j,0]/beta[i] -px[j,i]/beta[i]-px[j,0]/beta[0]
        if ak>np.log(np.random.random(1)):
            #changes posteriors
            p1=np.copy(px[j,i])
            p2=np.copy(px[j,0])
            px[j,i]=p2
            px[j,0]=p1
            
            #changes positions
            p1=np.copy(X[j,:,i])
            p2=np.copy(X[j,:,0])                
            X[j,:,i]=p2; 
            X[j,:,0]=p1; 
    elif strategy=='full_vanilla_forward':
        #swaps from top to bottom 
        for i in range(N_temp-1):
            #computes ak
            ak=px[j,i]/beta[i+1]+px[j,i+1]/beta[i]-px[j,i]/beta[i]-px[j,i+1]/beta[i+1]

            if ak>np.log(np.random.random(1)):
            #changes posteriors
                p1=np.copy(px[j,i])
                p2=np.copy(px[j,i+1])
                px[j,i]=p2
                px[j,i+1]=p1
                
                #changes positions
                p1=np.copy(X[j,:,i])
                p2=np.copy(X[j,:,i+1])                
                X[j,:,i]=p2; 
                X[j,:,i+1]=p1; 
    elif strategy=='probabilistic_vanilla_forward':
        #swaps from top to bottom 
        for i in range(N_temp-1):
            #computes ak
            if np.random.random(1)>0.5:
                ak=px[j,i]/beta[i+1]+px[j,i+1]/beta[i]-px[j,i]/beta[i]-px[j,i+1]/beta[i+1]
                if ak>np.log(np.random.random(1)):
                #changes posteriors
                    ####set_trace()
                    #changes posteriors
                    p1=np.copy(px[j,i])
                    p2=np.copy(px[j,i+1])
                    px[j,i]=p2
                    px[j,i+1]=p1
                    
                    #changes positions
                    p1=np.copy(X[j,:,i])
                    p2=np.copy(X[j,:,i+1])                
                    X[j,:,i]=p2; 
                    X[j,:,i+1]=p1; 

    

    elif strategy=='full_vanilla_backward':
        #print('------')
        #swaps from bottom to top
        ###set_trace()
        for i in range(N_temp-1,0,-1):
            #computes ak
            ak=px[j,i]/beta[i-1]+px[j,i-1]/beta[i]-px[j,i]/beta[i]-px[j,i-1]/beta[i-1]
            if ak>np.log(np.random.random(1)):
                #print('swapped '+str(i)+' and '+str(i-1))
                
                #if i==1: print(X[j,:,i-1]);print(X[j,:,i])
                
                #changes posteriors
                p1=np.copy(px[j,i])
                p2=np.copy(px[j,i-1])
                px[j,i]=p2
                px[j,i-1]=p1
                
                #changes positions
                p1=np.copy(X[j,:,i])
                p2=np.copy(X[j,:,i-1])                
                X[j,:,i]=p2; 
                X[j,:,i-1]=p1;  
                
                
                #if i==1: ###set_trace()/np.sum(pmat)   
    elif strategy=='full_random':
        
            #swaps two chains at random
            i,k=np.random.randint(0,N_temp,2)
                #computes ak
            ak=px[j,i]/beta[k]+px[j,k]/beta[i]-px[j,k]/beta[k]-px[j,i]/beta[i]
            if ak>np.log(np.random.random(1)):
                p1=np.copy(px[j,i])
                p2=np.copy(px[j,k])
                px[j,i]=p2
                px[j,k]=p1
                
                #changes positions
                p1=np.copy(X[j,:,i])
                p2=np.copy(X[j,:,k])                
                X[j,:,i]=p2; 
                X[j,:,k]=p1;  
            
    return X,px
    
    
    


def parallel_tempering(post,N,beta,sigma_rwm,x0,strategy,Ns=1,Disp=0):
    
    #preallocation:

    N_temp=len(beta)
    p=np.size(x0,0)

    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))

    # preallocates number of samnples
    X=np.zeros((N,p,N_temp))
    y=np.zeros((N_temp,p))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0


    #gets first posterior
    for i in range(N_temp):
        px[0,i]=post(X[0,:,i])

    #starts the main MCMC loop

    for j in range(N-1):

        if (np.mod(j,Ns)==0):
            #swaps according to a given strategy
            X,px=swap_strategy(X,px,beta,strategy,j)

    #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=X[j,:,k]+sigma_rwm[k]*np.random.standard_normal(p,)
            py[j,k]=post(y[k,:])

    #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]

        if Disp:
            display(X[j+1,:,0], acpt, j+1,px[j+1,0])
            
    print(acpt/N)
                       
    return X         



def full_vanilla(post,N,beta,sigma_rwm,x0,Ns=1,Disp=0):
    
    #preallocation:
    N_temp=len(beta)
    p=np.size(x0,0)

    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))

    # preallocates number of samnples
    X=np.zeros((N,p,N_temp))
    y=np.zeros((N_temp,p))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0


    #gets first posterior
    for i in range(N_temp):
        px[0,i]=post(X[0,:,i])

    #starts the main MCMC loop

    for j in range(N-1):

        if (np.mod(j,Ns)==0):
            #swaps according to a given strategy
            X,px=swap_strategy(X,px,beta,'full_vanilla_forward',j)
#                            

    #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=X[j,:,k]+sigma_rwm[k]*np.random.standard_normal(p,)
            py[j,k]=post(y[k,:])
        
        
        #print('before swap')
        #print(py)
        ####set_trace()
    #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
                #if i==0: print('accept')
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]
        ####set_trace()
        #swaps temperatures
        ###set_trace()
        if (np.mod(j,Ns)==0):
            #swaps according to a given strategy
            X,px=swap_strategy(X,px,beta,'full_vanilla_backward',j+1)
            #print('after swap')
            ####set_trace()
            
        
        if Disp:
            display(X[j+1,:,0], acpt, j+1,px[j+1,0])
            
            
    print(acpt/N)
    return X          



def pcn_full_vanilla(post,N,beta,rho,PRIOR,x0,Ns=1,Disp=0,dim_q=1):
    
    #preallocation:
    x0=np.float32(x0)

    N_temp=len(beta)
    p=len(x0[:,0])

    #prealocates posteriors
    px=np.zeros((N,N_temp))
    qoi=np.zeros((N,dim_q,N_temp))
    qoi_y=np.zeros((N,dim_q,N_temp))

    py=np.zeros((N,N_temp))

    # preallocates number of samnples
    XX=np.float32(np.zeros((N,p,N_temp)))
    y=np.float32(np.zeros((N_temp,p)))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    XX[0,:,:]=np.float32(x0)

    #creates matrix of shifts
    
    shift=(1-rho**2.0)**0.5

    #gets first posterior
    for i in range(N_temp):
        px[0,i],qoi[0,:,i]=post(XX[0,:,i])

    #starts the main MCMC loop

    for j in range(N-1):

        if (np.mod(j,Ns)==0):
            #swaps according to a given strategy
            XX,px,qoi=swap_strategy_pcn(XX,px,qoi,beta,'full_vanilla_forward',j)
#                            

    #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=shift[k]*XX[j,:,k]+rho[k]*PRIOR.sample(exp=False)
        
            py[j,k],qoi_y[j,:,k]=post(y[k,:])
        
        
        #print('before swap')
        #print(py)
    #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                XX[j+1,:,i]=np.copy(y[i,:])
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
                qoi[j+1,:,i]=qoi_y[j,:,i]
                #if i==0: print('accept')
            else:
                XX[j+1,:,i]=np.copy(XX[j,:,i])
                px[j+1,i]=px[j,i]
                qoi[j+1,:,i]=qoi[j,:,i]
        ####set_trace()
        #swaps temperatures
        if (np.mod(j,Ns)==0):
            #swaps according to a given strategy
            XX,px,qoi=swap_strategy_pcn(XX,px,qoi,beta,'full_vanilla_backward',j+1)
            #print('after swap')
            ####set_trace()
            
        
        if Disp:
            display('FULL PT',acpt, j+1,px[j+1,0])
            
            
    print(acpt/N)
    return XX,qoi      

           
"""


"""



"""
Creates the functions necesary for the infinite swapping algorithms,
both weighted and unweighted.

"""



"""
Computes the uw ratio as un equation (9) in the manuscript

"""


def compute_ratio_uw2(px,beta):
    L=list(it.permutations(np.arange(0,len(beta))))
    N_perm=len(L)
    per_px=list(it.permutations(px))
    p0_T=per_px/beta
    p0_T=np.sum(p0_T,1)
    r=np.zeros(N_perm)
    # computes weights from derivative of LSE
    # since the form of the weights is given by LSE
    h=0.0001
    LSE=logsumexp(p0_T)
    for i in range(N_perm):
        ph=p0_T
        ph[i]=p0_T[i]+h
        LSE_h=logsumexp(ph)
        r[i]=(LSE_h-LSE)/h
        ph[i]=p0_T[i]-h
    #rectifies weights
    r=r/np.sum(r)
    return r

def compute_ratio_uw(px,beta):
    eps=1E-730
    per_px=list(it.permutations(px))
    p0_T=per_px/beta
    p0_T=np.sum(p0_T,1)
    if any(np.exp(p0_T)>0.0):
        Sum=np.sum(np.exp(p0_T))+eps
        r=(np.exp(p0_T)+eps)/Sum
    else:
        r=compute_ratio_uw2(px,beta)
    r=r/np.sum(r)
    return r


# for the paper of Lacki 
def compute_swap_matrix(px):
    K=len(px)
    pmat=np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            ###set_trace()
            pmat[i,j]=np.exp(-np.abs(px[i]-px[j]))
    return pmat
    
def swap_matrix(px):
    K=len(px)
    pmat=compute_swap_matrix(px)
    pmat=pmat/np.sum(pmat)
    p_flat=pmat.flatten()     
    index_flat=np.random.choice(np.arange(K*K),p=p_flat)
    #finds index
    ###set_trace()
    aa=np.argwhere(pmat==p_flat[index_flat])
    return aa[0],pmat,p_flat,index_flat


def swap_states(px,beta):
    index,_,_,_=swap_matrix(px)
    i,j=index[0],index[1]
    bi=1/beta[i]
    bj=1/beta[j]
    aij=(bj-bi)*(px[i]-px[j])
    u=np.log(np.random.random(1)    )
    if u<aij:
        swap=1
    else:
        swap=0
    return swap,i,j

    
    
    
    
def state_dependent_PT(post,N,beta,sigma_rwm,x0,ts=1,Ns=1,Disp=0):
    #preallocation:
    Ndisp=int(0.1*N)
    N_temp=len(beta)
    p=np.size(x0,0)

    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))

    # preallocates number of samnples
    X=np.zeros((N,p,N_temp))
    y=np.zeros((N_temp,p))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0

    #gets first posterior
    for i in range(N_temp):

        px[0,i]=post(X[0,:,i])



    for j in range(N-1):

        #displays or not
        if Disp==1:
            if np.mod(j,Ndisp)==0:
                print('iteration '+str(j))
        
            
        #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=X[j,:,k]+sigma_rwm[k]*np.random.standard_normal(p)
            py[j,k]=post(y[k,:])
            

        #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]
                
        #Does the swap
                
        swap,I,J=swap_states(px[j+1,:], beta)
        if swap==1:
            xx=np.copy(X[j+1,:,I])
            pxx=np.copy(px[j+1,I])
            
            X[j+1,:,I]=np.copy(X[j+1,:,J])
            px[j+1,I]= np.copy(px[j+1,J])          
            
            X[j+1,:,J]=xx
            px[j+1,J]=pxx
        
        if Disp:
            display(X[j+1,:,0], acpt, j+1,px[j+1,0])
    
        
    print(acpt/N)

    return X    
    
    
    
def pcn_state_dependent_PT(post,N,beta,rho,PRIOR,x0,ts=1,Ns=1,Disp=0,dim_q=1):
    #preallocation:
    x0=np.float32(x0)
    Ndisp=int(0.1*N)
    N_temp=len(beta)
    p=np.size(x0,0)

    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))
    
    
    qoi=np.zeros((N,dim_q,N_temp))
    qoi_y=np.zeros((N,dim_q,N_temp))

    # preallocates number of samnples
    X=np.float32(np.zeros((N,p,N_temp)))
    y=np.float32(np.zeros((N_temp,p)))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0
    
    shift=(1-rho**2.0)**0.5

    #gets first posterior
    for i in range(N_temp):

        px[0,i],qoi[0,:,i]=post(X[0,:,i])



    for j in range(N-1):

        #displays or not
        if Disp==1:
            if np.mod(j,Ndisp)==0:
                print('iteration '+str(j))
        
            
        #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=shift[k]*X[j,:,k]+rho[k]*PRIOR.sample(exp=False)
            py[j,k],qoi_y[j,:,k]=post(y[k,:])
            

        #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
                qoi[j+1,:,i]=qoi_y[j,:,i]
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]
                qoi[j+1,:,i]=qoi[j,:,i]
                
        #Does the swap
                
        swap,I,J=swap_states(px[j+1,:], beta)
        if swap==1:
            xx=np.copy(X[j+1,:,I])
            pxx=np.copy(px[j+1,I])
            qxx=np.copy(qoi[j+1,:,I])
            
            X[j+1,:,I]=np.copy(X[j+1,:,J])
            px[j+1,I]= np.copy(px[j+1,J])          
            qoi[j+1,:,I]=np.copy(qoi[j+1,:,J])  
            X[j+1,:,J]=xx
            px[j+1,J]=pxx
            qoi[j+1,:,J]=qxx
        
        if Disp:
            display('SD', acpt, j+1,px[j+1,0])
    
        
    print(acpt/N)

    return X,qoi   
    
    





"""
swaps based on the uw ratio

"""



def swap(px,beta):

    r=compute_ratio_uw(px,beta)
    #computes CDF
    idx=np.arange(len(r))
    sigma=np.random.choice(idx,size=1,replace=False,p=r)
    sigma=sigma[0]
    return sigma,r

"""

Creates the base for unweighted IS

takes as input

 post: funciton evaluated at x
 N:    Number of samples
 beta: array of temperatures
 sigma_rwm: matrix of covariance
 x0:   initial value of length (N_temp,p)
 Disp optimal: displays progress

outputs:

X array of (N,P,N_temp) samples


"""



def unweighted_IS(post,N,beta,sigma_rwm,x0,ts=1,Ns=1,Disp=0):
    #preallocation:
    Ndisp=int(0.1*N)
    N_temp=len(beta)
    p=np.size(x0,0)


    perm_list=list(it.permutations(np.arange(0,N_temp)))

    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))

    # preallocates number of samnples
    X=np.zeros((N,p,N_temp))
    y=np.zeros((N_temp,p))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0
    acpt_s=0
    sigma=0
    #gets first posterior
    for i in range(N_temp):

        px[0,i]=post(X[0,:,i])



    for j in range(N-1):

        #displays or not
        if Disp==1:
            if np.mod(j,Ndisp)==0:
                print('iteration '+str(j))
        # Does the first swap
        sigma_=sigma
        if np.mod(j,Ns)==0 :
            sigma,_=swap(px[j,:],beta)
            X[j,:,:]=X[j,:,perm_list[sigma]].T
            px[j,:]=px[j,perm_list[sigma]]

            
        #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=X[j,:,k]+sigma_rwm[k]*np.random.standard_normal(p,)
            py[j,k]=post(y[k,:])

        #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]

        #Does the second swap?
                
        if ts:
            if np.mod(j,Ns)==0 :
                sigma,_=swap(px[j+1,:],beta)
                X[j+1,:,:]=X[j+1,:,perm_list[sigma]].T
                px[j+1,:]=px[j+1,perm_list[sigma]]
            if sigma_!=sigma:
                acpt_s+=1
                
                
        
        if Disp:
            display(X[j+1,:,0], acpt, j+1,px[j+1,0])
        
    print(acpt/N)

    return X







def pcn_unweighted_IS(post,N,beta,rho,PRIOR,x0,ts=1,Ns=1,Disp=0,dim_q=1,PLOT=False,folder=None):
    #preallocation:
    x0=np.float32(x0)
    Ndisp=int(0.1*N)
    N_temp=len(beta)
    p=len(x0[:,0])


    perm_list=list(it.permutations(np.arange(0,N_temp)))

    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))
    qoi=np.zeros((N,dim_q,N_temp))
    qoi_y=np.zeros((N,dim_q,N_temp))
    # preallocates number of samnples
    X=np.float32(np.zeros((N,p,N_temp)))
    y=np.float32(np.zeros((N_temp,p)))
    shift=(1-rho**2.0)**0.5
    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0
    acpt_s=0
    sigma=0
    #gets first posterior
    for i in range(N_temp):

        px[0,i],qoi[0,:,i]=post(X[0,:,i])



    for j in range(N-1):

        #displays or not
        if Disp==1:
            if np.mod(j,Ndisp)==0:
                print('iteration '+str(j))
        # Does the first swap
        sigma_=sigma
        if np.mod(j,Ns)==0 :
            sigma,_=swap(px[j,:],beta)
            X[j,:,:]=X[j,:,perm_list[sigma]].T
            px[j,:]=px[j,perm_list[sigma]]
            qoi[j,:,:]=qoi[j,:,perm_list[sigma]].T

            
        #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=shift[k]*X[j,:,k]+rho[k]*PRIOR.sample(exp=False)
            py[j,k],qoi_y[j,:,k]=post(y[k,:])

        #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            
            # import pdb
            # pdb.##set_trace()
            
            
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
                qoi[j+1,:,i]=qoi_y[j,:,i]
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]
                qoi[j+1,:,i]=qoi[j,:,i]


        #Does the second swap?
        
        if ts:
            if np.mod(j,Ns)==0 :
                sigma,_=swap(px[j+1,:],beta)
                X[j+1,:,:]=X[j+1,:,perm_list[sigma]].T
                px[j+1,:]=px[j+1,perm_list[sigma]]
                qoi[j+1,:,:]=qoi[j+1,:,perm_list[sigma]].T
            if sigma_!=sigma:
                acpt_s+=1
                
        if PLOT:
            import matplotlib
            matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
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
            field=X[j+1,:,0]
            v2f=dl.vertex_to_dof_map(V)
            Xgrid,Ygrid=np.meshgrid(np.linspace(0,3,29),np.linspace(0,2,29))
            
            N_rec_x=5
            rx=np.linspace(1,2,N_rec_x)
            ry=1.95*np.ones(len(rx))
            plt.figure(figsize=(9,4))
            vmin=-3.5
            vmax=3.5
            plt.contourf(Xgrid,Ygrid,field[v2f].reshape((29,29)),levels = np.linspace(vmin, vmax, 50+1),cmap='Spectral')
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.clim(vmin=-2.5,vmax=2.5)
            plt.colorbar()

            plt.savefig(folder+'/field_'+str(j).zfill(2)+'.png')

            #plt.draw()
            #plt.pause(0.0001)
            #plt.show()
                
        
        if Disp:
            display('uw', acpt, j+1,px[j+1,0])
        
    print(acpt/N)

    return X,qoi


def no_unweighted_IS(post,N,beta,rho,PRIOR,x0,ts=1,Ns=1,Disp=0):
    #preallocation:
    Ndisp=int(0.1*N)
    N_temp=len(beta)
    p=len(x0[:,0])


    perm_list=list(it.permutations(np.arange(0,N_temp)))

    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))
    qoi=np.zeros((N,1,N_temp))
    qoi_y=np.zeros((N,1,N_temp))
    # preallocates number of samnples
    X=np.zeros((N,p,N_temp))
    y=np.zeros((N_temp,p))
    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0
    acpt_s=0
    sigma=0
    #gets first posterior
    for i in range(N_temp):

        px[0,i],qoi[0,0,i]=post(X[0,:,i])



    for j in range(N-1):

        #displays or not
        if Disp==1:
            if np.mod(j,Ndisp)==0:
                print('iteration '+str(j))
        # Does the first swap
        sigma_=sigma
        if np.mod(j,Ns)==0 :
            sigma,_=swap(px[j,:],beta)
            X[j,:,:]=X[j,:,perm_list[sigma]].T
            px[j,:]=px[j,perm_list[sigma]]
            qoi[j,0,:]=qoi[j,0,perm_list[sigma]]

            
        #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=X[j,:,k]+rho[k]*PRIOR.sample(exp=False)
            py[j,k],qoi_y[j,0,k]=post(y[k,:])

        #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            
            # import pdb
            # pdb.##set_trace()
            
            
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
                qoi[j+1,0,i]=qoi_y[j,0,i]
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]
                qoi[j+1,0,i]=qoi[j,0,i]

        #Does the second swap?
                
        if ts:
            if np.mod(j,Ns)==0 :
                sigma,_=swap(px[j+1,:],beta)
                X[j+1,:,:]=X[j+1,:,perm_list[sigma]].T
                px[j+1,:]=px[j+1,perm_list[sigma]]
                qoi[j+1,0,:]=qoi[j+1,0,perm_list[sigma]]
            if sigma_!=sigma:
                acpt_s+=1
                
                
        
        if Disp:
            display('uw', acpt, j+1,px[j+1,0])
        
    print(acpt/N)

    return X,qoi

"""


Computes the weights for the weighted IS algorithm as in eqn (10)



"""




def compute_weights(px,beta):
    eps=1E-730
    per_beta=list(it.permutations(beta))
    p0_T=px/per_beta
    p0_T=np.sum(p0_T,1)
    if any(np.exp(p0_T)>0.0):
        Sum=np.sum(np.exp(p0_T))+eps
        r=(np.exp(p0_T)+eps)/Sum
    else:
        r=compute_weights2(px,beta)
    #rectifies
    r=r/np.sum(r)
    return r



def compute_weights2(px,beta):

    per_beta=list(it.permutations(beta))
    N_perm=len(per_beta)

    p0_T=px/per_beta
    p0_T=np.sum(p0_T,1)
    r=np.zeros(N_perm)
    # computes weights from derivative of LSE
    # since the form of the weights is given by LSE
    h=0.0001
    LSE=logsumexp(p0_T)
    for i in range(N_perm):
        ph=p0_T
        ph[i]=p0_T[i]+h
        LSE_h=logsumexp(ph)
        r[i]=(LSE_h-LSE)/h
        ph[i]=p0_T[i]-h

    #rectifies weights

    r=r/np.sum(r)
    return r

def swap_weighted(px,beta):

    w=compute_weights(px,beta)
    # computes CDF
    #cdf=np.cumsum(w)
    #samples weight
    #u=np.random.random(1)
    #sigma=np.argmax(cdf>u)
    idx=np.arange(len(w))
    sigma=np.random.choice(idx,size=1,replace=False,p=w)
    sigma=sigma[0]
    return sigma,w


"""

Creates the base for weighted IS

takes as input

 post: funciton evaluated at x
 N:    Number of samples
 beta: array of temperatures
 sigma_rwm: matrix of covariance
 x0:   initial value of length (N_temp,p)
 Disp optimal: displays progress

outputs:

X array of (N,P,N_temp) samples


"""





def weighted_IS(post,N,beta_original,sigma_rwm,PRIOR,x0,Disp=0):
    #preallocation:
    x0=np.float32(x0)
    Ndisp=int(0.1*N)            # display progress at every tenth
    N_temp=len(beta_original)   #number of temperates
    p=np.size(x0,0)             #number of parameters

    #permutation list
    perm_list=list(it.permutations(np.arange(0,N_temp)))   
    #number of permutations
    N_perm=len(perm_list)      

    #preallocates weights
    W_IS=np.zeros((N,N_perm))


    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))

    # preallocates number of samnples
    X=np.float32(np.zeros((N,p,N_temp)))
    y=np.float32(np.zeros((N_temp,p)))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0
    w=np.zeros((N,N_perm))
    #gets first posterior
    for i in range(N_temp):
        px[0,i]=post(X[0,:,i])
        
    beta=beta_original
    #gets first weights for importance sampling
    W_IS[0,:]=compute_ratio_uw(px[0,:],beta_original)
    #starts the main MCMC loop
    sigma_=0
    acpt_swap=0
    for j in range(N-1):

        #displays or not
        if Disp==1:
            if np.mod(j,Ndisp)==0:
                print('iteration '+str(j))


        #Does the wIS
        sigma, w[j,:]=swap_weighted(px[j,:],beta_original)
        
        if sigma!=sigma_:
            acpt_swap+=1
        sigma_=sigma

        index=perm_list[sigma]
        beta=beta_original[[index]]

    #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=X[j,:,k]+sigma_rwm[index[k]]*PRIOR.sample(exp=True)
            py[j,k]=post(y[k,:])

    #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                acpt[i]=acpt[i]+1
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]

        W_IS[j+1,:]=compute_ratio_uw(px[j+1,:],beta_original)
       
        
        if Disp:
            display(X[j+1,:,0], acpt, j+1,px[j+1,0])
            
            
            
    print(acpt/N)  
    print(acpt_swap/N)
    return X,W_IS,w




def pcn_weighted_IS(post,N,beta_original,rho,PRIOR,x0,Disp=0,dim_q=1):
    #preallocation:

    Ndisp=int(0.1*N)            # display progress at every tenth
    N_temp=len(beta_original)   #number of temperates
    p=len(x0[:,0])

    #permutation list
    perm_list=list(it.permutations(np.arange(0,N_temp)))   
    #number of permutations
    N_perm=len(perm_list)      

    #preallocates weights
    W_IS=np.zeros((N,N_perm))


    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))
    
    qoi=np.zeros((N,dim_q,N_temp))
    qoi_y=np.zeros((N,dim_q,N_temp))

    # preallocates number of samnples
    X=np.zeros((N,p,N_temp))
    y=np.zeros((N_temp,p))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0
    w=np.zeros((N,N_perm))
    #gets first posterior
    for i in range(N_temp):
        px[0,i],qoi[0,:,i]=post(X[0,:,i])
        
    beta=beta_original
    #gets first weights for importance sampling
    W_IS[0,:]=compute_ratio_uw(px[0,:],beta_original)
    #starts the main MCMC loop
    sigma_=0
    acpt_swap=0
    shift=(1-rho**2.0)**0.5
    for j in range(N-1):

        #displays or not
        if Disp==1:
            if np.mod(j,Ndisp)==0:
                print('iteration '+str(j))


        #Does the wIS
        sigma, w[j,:]=swap_weighted(px[j,:],beta_original)
        
        if sigma!=sigma_:
            acpt_swap+=1
        sigma_=sigma

        index=perm_list[sigma]
        beta=beta_original[[index]]

    #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=shift[index[k]]*X[j,:,k]+rho[index[k]]*PRIOR.sample(exp=False)
            py[j,k],qoi_y[j,:,k]=post(y[k,:])
            

    #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                qoi[j+1,:,i]=qoi_y[j,:,i]
                acpt[i]=acpt[i]+1
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]
                qoi[j+1,:,i]=qoi[j,:,i]

        W_IS[j+1,:]=compute_ratio_uw(px[j+1,:],beta_original)
       
        
        if Disp:
            display('w', acpt, j+1,px[j+1,0])
            
            
            
    print(acpt/N)  
    print(acpt_swap/N)
    return X,qoi,W_IS,w


def no_weighted_IS(post,N,beta_original,rho,PRIOR,x0,Disp=0):
    #preallocation:

    Ndisp=int(0.1*N)            # display progress at every tenth
    N_temp=len(beta_original)   #number of temperates
    p=len(x0[:,0])

    #permutation list
    perm_list=list(it.permutations(np.arange(0,N_temp)))   
    #number of permutations
    N_perm=len(perm_list)      

    #preallocates weights
    W_IS=np.zeros((N,N_perm))


    #prealocates posteriors
    px=np.zeros((N,N_temp))
    py=np.zeros((N,N_temp))
    
    qoi=np.zeros((N,1,N_temp))
    qoi_y=np.zeros((N,1,N_temp))

    # preallocates number of samnples
    X=np.zeros((N,p,N_temp))
    y=np.zeros((N_temp,p))

    #stores ratios and ar
    ratio=np.zeros(N_temp)
    acpt=np.zeros(N_temp)
    X[0,:,:]=x0
    w=np.zeros((N,N_perm))
    #gets first posterior
    for i in range(N_temp):
        px[0,i],qoi[0,0,i]=post(X[0,:,i])
        
    beta=beta_original
    #gets first weights for importance sampling
    W_IS[0,:]=compute_ratio_uw(px[0,:],beta_original)
    #starts the main MCMC loop
    sigma_=0
    acpt_swap=0
    shift=(1-rho**2.0)**0.5
    for j in range(N-1):

        #displays or not
        if Disp==1:
            if np.mod(j,Ndisp)==0:
                print('iteration '+str(j))


        #Does the wIS
        sigma, w[j,:]=swap_weighted(px[j,:],beta_original)
        
        if sigma!=sigma_:
            acpt_swap+=1
        sigma_=sigma

        index=perm_list[sigma]
        beta=beta_original[[index]]

    #computes each individual posterior
        for k in range(N_temp):
            y[k,:]=X[j,:,k]+rho[index[k]]*PRIOR.sample(exp=False)
            py[j,k],qoi_y[j,0,k]=post(y[k,:])
            

    #accepts-rejects
        for i in range(N_temp):
            ratio[i]=py[j,i]/beta[i]-px[j,i]/beta[i];
            if (np.log(np.random.random(1))<ratio[i]):
                X[j+1,:,i]=y[i,:]
                px[j+1,i]=py[j,i]
                qoi[j+1,0,i]=qoi_y[j,0,i]
                acpt[i]=acpt[i]+1
            else:
                X[j+1,:,i]=X[j,:,i]
                px[j+1,i]=px[j,i]
                qoi[j+1,0,i]=qoi[j,0,i]

        W_IS[j+1,:]=compute_ratio_uw(px[j+1,:],beta_original)
       
        
        if Disp:
            display('w', acpt, j+1,px[j+1,0])
            
            
            
    print(acpt/N)  
    print(acpt_swap/N)
    return X,qoi,W_IS,w

def weight_samples(x,w,N_temp=None):

    N=np.size(x,0)
    Np=np.size(x,1)
    if N_temp is  None:
        N_temp=np.size(x,2)
        
    perm_list=list(it.permutations(np.arange(0,N_temp)))
    N_perm=len(perm_list)   #number of permutations

    Y=np.zeros((N*N_perm,Np,N_temp))
    W=np.zeros((N*N_perm))

    for i in range(N):
        W[i*N_perm:N_perm*(i+1)]=w[i,:]
        xx=np.asarray(list(it.permutations(x[i,:,:].T)))
        xx=np.transpose(xx,[0,2,1])
        Y[i*N_perm:N_perm*(i+1),:,:]=xx

    return Y,W

def resample_IS(x,w,N):
    indx=np.arange(len(x))
    # reweights just in case
    w=w/sum(w)
    indx=np.random.choice(indx,N,replace=True,p=w)
    return x[indx]

def weight(x,w):
    N=x.shape[0]
    p=x.shape[1]
    N_temp=x.shape[2]
    perm_list=np.asarray(list(it.permutations(np.arange(N_temp))))
    xx=np.zeros((N,p))
    
    for i in range(N):
        for j in range(N_temp):
            ww=np.sum(w[i,perm_list[:,j]==0])
            xx[i,:]+=x[i,:,j]*ww
    return xx
    


def weight3(x,w):
    N=x.shape[0]
    p=x.shape[1]
    N_temp=x.shape[2]
    perm_list=np.asarray(list(it.permutations(np.arange(N_temp))))
    ww=np.zeros((N,N_temp))
    for i in range(N):
        for j in range(N_temp):
            ww[i,j]=np.sum(w[i,perm_list[:,j]==0])
        
    return x,ww


def resample(x,w,N):
    y=np.zeros((N,2))
    for i in range(N):
        j=np.random.choice(np.arange(4),size=1,p=w[i,:])
        y[i,:]=x[i,:,j]
    return y

def mean(x,w):
    xx=weight(x,w)
    return np.mean(xx,0)

def weight2(x,w):
    N=x.shape[0]
    p=x.shape[1]
    N_temp=x.shape[2]
    perm_list=np.asarray(list(it.permutations(np.arange(N_temp))))
    xx=np.zeros((N,p))
    
    for i in range(N):
        for j in range(N_temp):
            ww=np.sum(w[i,perm_list[:,0]==j])
            xx[i,:]+=x[i,:,j]*ww
    return xx
    


def weights_jonas(X,weight,t=0):
    
    N_temp=X.shape[2]
    
    
    Y0 = X[:,0,t];
    Y1 = X[:,1,t];
    W = weight[:,0];
    List_perm=np.asarray(list(it.permutations(np.arange(N_temp))))

    for i in range(1,np.math.factorial(N_temp)):
        Perm = List_perm[i];
        #print(Perm)
        Y0 = np.append(Y0, X[:,0,Perm[t]])
        Y1 = np.append(Y1, X[:,1,Perm[t]])
        W = (1./np.math.factorial(N_temp))*np.append(W, weight[:,i])
            
    return np.array([Y0,Y1]),W
    

def mean_jonas(x,w):
    yj,wj=weights_jonas(x,w)
    return np.average(yj,1,weights=wj)

