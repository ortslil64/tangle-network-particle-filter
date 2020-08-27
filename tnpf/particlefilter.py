#!/usr/bin/env python3

import numpy as np
from scipy.stats import multivariate_normal, norm

class particle_filter():
    def __init__(self, X0, P0, R,Q , Np = 100, verbos = False):
        self.verbos = verbos
        self.Np = Np
        self.R = R
        self.Q = Q
        self.P = P0
        self.W = np.ones(Np)/Np
        self.X = np.random.normal(X0, np.sqrt(P0), Np)
        
    def predict(self, dt):
        self.X = np.random.normal(np.sin(dt*self.X), 1.5*np.sqrt(self.Q), self.Np)
    
    def update(self, z):
        self.W = self.W * norm(loc = z, scale = np.sqrt(self.R)).pdf(self.X)
        if np.sum(self.W) == 0:
            self.W = np.ones(self.Np)/self.Np
        else:
            self.W = self.W/np.sum(self.W)
        
        Neff = 1/(np.sum(np.power(self.W, 2)))
        if Neff < (2/3)*self.Np:
            if self.verbos:
                print("Performing resample...")
            idxs = np.random.choice(a = self.Np,size = self.Np, p = self.W)
            self.X = self.X[idxs]
            self.X = np.random.normal(self.X, 0.1*np.sqrt(self.Q), self.Np)
            self.W = np.ones(self.Np)/self.Np
            if self.verbos:
                print("Done resample!")
    
    def estimate(self):
        return self.X.T.dot(self.W)
    
    
    
class centrlized_particle_filter():
    def __init__(self, X0, P0, R,Q , Np = 100, verbos = False):
        self.verbos = verbos
        self.Np = Np
        self.R = R
        self.Q = Q
        self.P = P0
        self.W = np.ones(Np)/Np
        self.X = np.random.normal(X0, np.sqrt(P0), Np)
        
    def predict(self, dt):
        self.X = np.random.normal(np.sin(dt*self.X), 1.5*np.sqrt(self.Q), self.Np)
    
    def update(self, z):
        self.W = self.W * multivariate_normal(mean = z, cov = np.diag(self.R)).pdf(np.repeat([self.X], len(z), axis=0).T)
        if np.sum(self.W) == 0:
            self.W = np.ones(self.Np)/self.Np
        else:
            self.W = self.W/np.sum(self.W)
        
        Neff = 1/(np.sum(np.power(self.W, 2)))
        if Neff < (2/3)*self.Np:
            if self.verbos:
                print("Performing resample...")
            idxs = np.random.choice(a = self.Np,size = self.Np, p = self.W)
            self.X = self.X[idxs]
            self.X = np.random.normal(self.X, 0.1*np.sqrt(self.Q), self.Np)
            self.W = np.ones(self.Np)/self.Np
            if self.verbos:
                print("Done resample!")
    
    def estimate(self):
        return self.X.T.dot(self.W)
            
        