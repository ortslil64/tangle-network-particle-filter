#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
import time
import multiprocessing as mp
from sklearn import mixture


class GMM():
    def __init__(self, Mus, Sigmas, Weights):
        self.Mus = Mus
        self.Sigmas = Sigmas
        self.Weights = Weights
        self.components = len(Mus)
    def pdf(self, x):
        f = np.zeros_like(x)
        for ii in range(self.components):
            f = f + self.Weights[ii]*norm(loc = self.Mus[ii], scale = self.Sigmas[ii]).pdf(x)
        return f

def get_gmm_from_pf(pf, n_components):
    s = np.random.choice(pf.Np, pf.Np, p = pf.W)
    X = pf.X[s]
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full').fit(X.reshape(-1, 1))
    return gmm

def gmm_worker(arg):
    pfs, ii ,n_components = arg
    gmm = get_gmm_from_pf(pfs[ii],n_components)
    return gmm

def get_fuzed_prob(x, gmms, A):
    f = 1
    for ii in range(len(gmms)):
        f = f * (np.exp(gmms[ii].score(np.array(x).reshape(-1, 1)))**A[ii])
    return f

def matropolis_hasting(pf, gmms, A):
    new_particles = np.zeros_like(pf.X)
    x = pf.X[0]
    w = get_fuzed_prob(x, gmms, A)
    if w == 0 or np.isnan(w) == True:
        w = 1/pf.Np
    for jj in range(pf.Np):
        s_t = np.random.choice(pf.Np)
        x_t = pf.X[s_t]
        w_t = get_fuzed_prob(x_t, gmms, A)
        if w_t == 0 or np.isnan(w_t) == True:
            w_t = 1/pf.Np
        if w_t > w:
            new_particles[jj] = x_t
            w = w_t
            x = x_t
        elif np.random.binomial(1, w_t/w) == 1:
            new_particles[jj] = x_t
            w = w_t
            x = x_t
        else:
            new_particles[jj] = x
    return new_particles
    
class DPF():
    def __init__(self, Na, n_components, A = None):
        self.Na = Na
        self.n_components = n_components
        if A is None:
            A = np.random.rand(Na)
            A = A / A.sum()
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
        gmms = pool.map(gmm_worker, ((pfs, ii, self.n_components) for ii in range(self.Na)))
        pool.close()
        pool.join()
        
        for ii in range(self.Na):
            pfs[ii].X = matropolis_hasting(pfs[ii], gmms, self.A)
            pfs[ii].W = np.ones_like(pfs[ii].W)/pfs[ii].Np  

                
        dt = time.time() - t0
        return pfs, dt
    
        