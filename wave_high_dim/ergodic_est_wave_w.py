#---------------------------------------------------------------------
import numpy as np
import base as base
import time
import sys
import prior2 as prior
import dolfin as dl
import forward_wave as et
import math
#sys.path.insert(0, '../../src')
import sys
import time
import pdb
import prior2 as prior
#sys.path.insert(0, '../../src')
dl.set_log_active(False)

INDICATOR=int(sys.argv[1])
print('Seed')
print(INDICATOR)


print('FPT')
#------------------------------------------------------
#
#   Sets up the problem
#
#------------------------------------------------------




data=np.load('data_wave.npy')
noise=np.load('noise_wave.npy')
theta_true=np.load('true_field.npy')

dt=0.0007
T=0.6
def misfit(x,data,noise):
    Nr=data.shape[1]
    f,Q=et.forward(x)
    log_likelihood=0
    for i in range(Nr):
        log_likelihood+=np.sum((data[:,i]-f[:,i])**2.0)*dt

    log_likelihood=-0.5*log_likelihood/(noise**2.0)/Nr
    return log_likelihood,Q



def log_post(x,data,noise):
    log_likelihood,Q=misfit(x,data,noise)
    #log_prior=-0.5*np.sum(x**2)
    log_post= log_likelihood#+log_prior
    return log_post,Q

post= lambda u: log_post(u,data,noise)


#%%
#---------------------------------------------------------------------
#
# Defines some hyper parameters, Number of temperatures, N samples, etc.
#
#---------------------------------------------------------------------

# Number of runs
Nruns=1
N=int(4*10**3); #number of samples
N_temp=4;#number of temperatures
Ns=1; #How often do we swap



Nx=28
Ny=28
mesh = dl.UnitSquareMesh(Nx, Ny)
PRIOR=prior.prior_measure(mesh)
z0=PRIOR.sample(exp=False)


p=len(z0)
X=np.zeros([N,p,N_temp])#prealloactes
y=np.zeros([N_temp,p]); #preallocates proposals
bt=np.zeros([N,N_temp])
T0=100.0**(1.0/3.0)
beta = np.array(T0**np.arange(0,N_temp));
beta_original = np.copy(beta);
x0=np.zeros((p,N_temp))

for i in range(N_temp):
    x0[:,i]=PRIOR.sample(exp=False)

#Rho for the pCN

sigma_is=np.array([[0.1],[0.2], [0.6],[1]]);#*linspace(1,100,N_temp);


dim_q=7

mean_r=np.zeros((Nruns,dim_q))
mean_p=np.zeros((Nruns,dim_q))
mean_uw=np.zeros((Nruns,dim_q))
mean_w=np.zeros((Nruns,dim_q))
mean_y=np.zeros((Nruns,dim_q))
mean_z=np.zeros((Nruns,dim_q))
mean_pf=np.zeros((Nruns,dim_q))
mean_sd=np.zeros((Nruns,dim_q))
#defines burn in
Burn_in=int(np.ceil(0.2*N))
N=N+Burn_in
Xsd=np.zeros((N,dim_q))
Xptf=np.zeros((N,dim_q))
Xuw=np.zeros((N,dim_q))
Xw=np.zeros((N,dim_q))
Xrw=np.zeros((N,dim_q))
Xsd=np.zeros((N,dim_q))


#---------------------------------------------------------------------
#
# Runs the algorithm Nrun times and compute the expected value of each run
#
print('started run of ergodic estimator...')
print('Number of samples ' +str(N))
t0=time.time()
for i in range(Nruns):
    x0=np.random.random((p,N_temp))

    print('----------------------------------')
    print('iteration '+str(i))

    """
    state dependent PT

    """

    # _,Xsd=base.pcn_state_dependent_PT(post,N,beta,sigma_is,x0,Ns=1,Disp=0) #State dependent
    # mean_sd[i,:]=np.mean(Xsd[Burn_in:,:,0],0)


    """
    Full PT

    """

    # Zptf,Xptf=base.pcn_full_vanilla(post,N,beta,sigma_is,x0,Ns=1,Disp=1) #full vanilla, reversible
    # mean_pf[i,:]=np.mean(Xptf[Burn_in:,:,0],0)

    """
    Un-Weighted

    """


    # Zuw,Xuw=base.pcn_unweighted_IS(post,N,beta,sigma_is,x0,1,1,Disp=0) #unweighted Is
    # mean_uw[i,:]=np.mean(Xuw[Burn_in:,:,0],0)


    """
    Weighted

    """


    Zw,Xw,W_IS,_=base.pcn_weighted_IS(post,N,beta_original,sigma_is,PRIOR,x0,Disp=1,dim_q=7) #weighted IS
    yy,ww=base.weight_samples(Xw,W_IS,N_temp=4)
    xx=base.resample_IS(yy,ww,N)
    mean_y[i,:]=np.mean(xx[Burn_in:,:,0],0)

    """
    pCN

    """

    # Zpcn,Xrwm=base.pcn(post,N*N_temp,sigma_is[0],x0[:,0],Disp=0) #random walk metropolis
    # mean_r[i,:]=np.mean(Xrwm[N_temp*Burn_in:],0)





name_y='results/est_y_'+str(INDICATOR).zfill(2)+'.npy'
name_s='results/samp_w_'+str(INDICATOR).zfill(2)+'.npy'



np.save(name_y,mean_y)



np.save(name_s,Zw)
