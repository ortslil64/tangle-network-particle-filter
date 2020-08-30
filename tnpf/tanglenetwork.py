#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm, multivariate_normal
import time
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors



class GMM():
    def __init__(self, Mus, Sigmas, Weights):
        self.Mus = Mus
        self.Sigmas = Sigmas
        self.Weights = Weights
        self.components = len(Mus)
    def pdf(self, x):
        f = np.zeros(len(x))
        for ii in range(self.components):
            f = f + self.Weights[ii]*multivariate_normal(mean = self.Mus[ii,:], cov = self.Sigmas[ii]).pdf(x)
        return f

def get_gmm_from_pf(pf, sigma):
    Mus = pf.X
    Weights = pf.W
    Sigmas = [np.eye(len(pf.Q)) * sigma]*len(pf.W)
    return GMM(Mus, Sigmas, Weights)


def worker(arg):
    w = []
    pfs, ii ,sigma = arg
    gmm = get_gmm_from_pf(pfs[ii],sigma)
    for jj in range(len(pfs)):
        if ii == jj:
            w.append(pfs[ii].W)
        else:
            w_temp = gmm.pdf(pfs[jj].X)
            #w_temp = w_temp/w_temp.sum()
            w.append(w_temp)
    return w
    
    
class tangle_network():
    def __init__(self, Na, sigma, A = None):
        self.Na = Na
        self.sigma = sigma
        if A is None:
            A = np.random.rand(Na,Na)
            A = A / A.sum(axis = 1)[:,None]
            self.A = A
        else:
            self.A = A
    
    def get_fusion_params(self, pfs, z):
        w = np.zeros(len(pfs))
        for ii in range(len(pfs)):
            w[ii] = (np.linalg.norm(pfs[ii].estimate() - z))
        w = w/w.sum()
        for ii in range(len(pfs)):
            for jj in range(len(pfs)):
                self.A[ii,jj] = w[jj]/w[ii]
        self.A = self.A / self.A.sum(axis = 1)[:,None]
                
    def fuse_particle_filters(self, pfs, n_workers = None):
        t0 = time.time()
        pfs_weights = np.empty((self.Na,self.Na), dtype=object)
        if n_workers is None:
            pool = mp.Pool(mp.cpu_count())
        else:
            pool = mp.Pool(n_workers)
        pfs_weights = pool.map(worker, ((pfs, ii, self.sigma) for ii in range(self.Na)))
        pool.close()
        pool.join()
        
                
        for ii in range(self.Na):
            w = np.array([x for x in pfs_weights[:][ii]], dtype=np.float64)
            alpha = self.A[ii,:]
            w = w ** alpha[:, None]
            w = np.prod(w, axis=0)
            if np.sum(w) == 0:
                w = np.ones_like(w)/pfs[ii].Np
            else:
                w = w/np.sum(w)
            pfs[ii].W = w
            pfs[ii].resample
        dt = time.time() - t0
        return pfs, dt
    
 
def sensors_pose2fusion_mat(poses, k_groups):
    nbrs = NearestNeighbors(n_neighbors=k_groups, algorithm='ball_tree').fit(poses)
    A = nbrs.kneighbors_graph(poses).toarray()
    A = A / A.sum(axis = 1)[:,None]
    return A
    
    

def log_fuzer_worker(arg):
    pfs, ii, A = arg
    Np = pfs[ii].Np
    x_temp = []
    for jj in range(len(A)):
        alpha = A[ii,jj]
        w = pfs[jj].W 
        s = np.random.choice(pfs[jj].Np, int(np.ceil(alpha*Np)), p=w)
        x_temp.append(pfs[jj].X[s])
    pfs[ii].X = np.concatenate(x_temp, axis = 0)
    pfs[ii].X = pfs[ii].X[:Np]
    pfs[ii].W = np.ones_like(pfs[ii].W)/pfs[ii].Np
    return pfs[ii]


class log_tangle_network():
    def __init__(self, Na, sigma, A = None):
        self.Na = Na
        self.sigma = sigma
        if A is None:
            A = np.random.rand(Na,Na)
            A = A / A.sum(axis = 1)[:,None]
            self.A = A
        else:
            self.A = A
    
    def get_fusion_params(self, pfs, z):
        w = np.zeros(len(pfs))
        for ii in range(len(pfs)):
            w[ii] = (np.linalg.norm(pfs[ii].estimate() - z))
        w = w/w.sum()
        for ii in range(len(pfs)):
            for jj in range(len(pfs)):
                self.A[ii,jj] = w[jj]/w[ii]
        self.A = self.A / self.A.sum(axis = 1)[:,None]
                
    def fuse_particle_filters(self, pfs, n_workers = None):
        t0 = time.time()
           
        if n_workers is None:
            pool = mp.Pool(mp.cpu_count())
        else:
            pool = mp.Pool(n_workers)
        pfs = pool.map(log_fuzer_worker, ((pfs, ii, self.A) for ii in range(self.Na)))
        pool.close()
        pool.join()
        dt = time.time() - t0
        return pfs, dt
    
        