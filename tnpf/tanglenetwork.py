#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
import time
import multiprocessing as mp



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

def get_gmm_from_pf(pf, sigma):
    Mus = pf.X
    Weights = pf.W
    Sigmas = np.ones_like(Weights) * sigma
    return GMM(Mus, Sigmas, Weights)

def worker(arg):
    w = []
    pfs, ii ,sigma = arg
    gmm = get_gmm_from_pf(pfs[ii],sigma)
    for jj in range(len(pfs)):
        w.append(gmm.pdf(pfs[jj].X))
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
                
    def fuse_particle_filters(self, pfs):
        t0 = time.time()
        pfs_weights = np.empty((self.Na,self.Na), dtype=object)
        
        pool = mp.Pool(mp.cpu_count())
        pfs_weights = pool.map(worker, ((pfs, ii, self.sigma) for ii in range(self.Na)))
        pool.close()
        pool.join()
        
        
        # for ii in range(self.Na):
        #     gmm = get_gmm_from_pf(pfs[ii],self.sigma)
        #     for jj in range(self.Na):
        #         pfs_weights[ii,jj] = gmm.pdf(pfs[jj].X)
                
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
        dt = time.time() - t0
        return pfs, dt
    
        