#!/usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal, norm

def process_model_sin(x, dt, Q):
    next_x = np.random.normal(np.sin(dt*x), np.sqrt(Q))
    return next_x

def likelihood_model_sin(x, z, R):
    if len(z) == 1:
        L = norm(loc = z[0], scale = np.sqrt(R[0])).pdf(x)
    else:
        L = multivariate_normal(mean = z, cov = R).pdf(np.repeat([x], len(z), axis=0).T)
    return L

def mesurment_model_sin(x, R):
    Nz = len(R)
    X = np.ones(Nz)*x
    e = np.random.multivariate_normal(np.zeros(Nz), R)
    z = X+e
    return z

def process_model_fly(x, dt, Q, v , omega):
    next_x = np.zeros_like(x)
    eps = np.random.multivariate_normal(np.zeros(x.shape[1]), Q, x.shape[0])
    next_x[:,0] = x[:,0] + dt*v*np.cos(x[:,2]) + eps[:,0]
    next_x[:,1] = x[:,1] + dt*v*np.sin(x[:,2]) + eps[:,1]
    next_x[:,2] = x[:,2] + dt*omega + eps[:,2]
    return next_x
    
def likelihood_model_fly(x,pose, z, R):
    rel_x = x[:,0:2] - pose
    z_hat = np.array([np.linalg.norm(rel_x, axis=1), np.arctan2(rel_x[:,0], rel_x[:,1])])
    L = multivariate_normal(mean = z[0], cov = R[0]).pdf(z_hat.T) 
    return L
    
def mesurment_model_fly(x, poses, R):
    Nz = len(poses)
    z = np.zeros((Nz, 2))
    for ii in range(Nz):
        eps = np.random.multivariate_normal(np.zeros(2), R[ii])
        rel_x = x[0:2] - poses[ii]
        z[ii,0] = np.linalg.norm(rel_x) + eps[0]
        z[ii,1] = np.arctan2(rel_x[0], rel_x[1]) + eps[1]
    return z