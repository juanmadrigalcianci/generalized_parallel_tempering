#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:45:58 2021

@author: juan
"""


import numpy as np
import dolfin as dl 
import matplotlib.pyplot as plt
from forward_wave import forward
import time
dl.set_log_active(False)



def th_true():
    return np.load('true_field.npy')


def generate_data(te):
    #te=th_true()
    y_true,Qt=forward(te)
    np.save('y_true_wave.npy',y_true)
    return y_true,Qt,te

def pollute_data(te):
    
    y_true,Qt,tt=generate_data(te)
    noise=np.max(y_true)*0.005
    data=y_true+noise*np.random.standard_normal(y_true.shape)
    np.save('data_wave.npy',data)
    np.save('noise_wave.npy',noise)
    return data,y_true,noise,tt


def misfit(x,l,data,noise):
    Nr=data.shape[1]
    f,Q=forward(x,l)
    log_likelihood=-0.5*np.linalg.norm(data-f)**2.0/(noise**2.0)/Nr
    return log_likelihood,Q
    
    

def log_post(x,l,data,noise):
    log_likelihood,Q=misfit(x,l,data,noise)
    #log_prior=-0.5*np.sum(x**2)    
    log_post= log_likelihood#+log_prior
    return log_post,Q


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