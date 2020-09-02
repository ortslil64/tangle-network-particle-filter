#!/usr/bin/env python3

import numpy as np
from scipy.stats import multivariate_normal, norm
from models import process_model_sin, likelihood_model_sin, process_model_fly, likelihood_model_fly

class particle_filter_sin():
    def __init__(self, X0, P0, R ,Q , Np = 100, verbos = False):
        self.verbos = verbos
        self.Np = Np
        self.R = R
        self.Q = Q
        self.P = P0
        self.W = np.ones(Np)/Np
        self.X = np.random.normal(X0, np.sqrt(P0), Np)
        
    def predict(self, dt):
        self.X = process_model_sin(self.X, dt, self.Q)
        
    def update(self, z):
        self.W = self.W * likelihood_model_sin(self.X, [z], [self.R])
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
    

class particle_filter_fly():
    def __init__(self, X0, P0, R ,Q , pose, v, omega, Np = 100, verbos = False):
        self.verbos = verbos
        self.Np = Np
        self.R = R
        self.Q = Q
        self.P = P0
        self.pose = pose
        self.v = v
        self.omega = omega
        self.W = np.ones(Np)/Np
        self.X = np.random.multivariate_normal(X0, P0, Np)
        
    def predict(self, dt):
        self.X = process_model_fly(self.X, dt, 1.5*self.Q, self.v, self.omega)
    
    def resample(self):
        idxs = np.random.choice(a = self.Np,size = self.Np, p = self.W)
        self.X = self.X[idxs]
        self.W = np.ones(self.Np)/self.Np
        
    def update(self, z):
        self.W = self.W * likelihood_model_fly(self.X, self.pose, [z], [self.R])
        if np.sum(self.W) == 0:
            self.W = np.ones(self.Np)/self.Np
        else:
            self.W = self.W/np.sum(self.W)
        
        Neff = 1/(np.sum(np.power(self.W, 2)))
        if Neff < (2/3)*self.Np:
            if self.verbos:
                print("Performing resample...")
            self.resample()
            if self.verbos:
                print("Done resample!")
    
    def estimate(self):
        return self.X.T.dot(self.W)
    
    


class centrlized_particle_filter_fly():
    def __init__(self, X0, P0, R ,Q , pose, v, omega, Np = 100, verbos = False):
        self.verbos = verbos
        self.Np = Np
        self.R = R
        self.Q = Q
        self.P = P0
        self.pose = pose
        self.v = v
        self.omega = omega
        self.W = np.ones(Np)/Np
        self.X = np.random.multivariate_normal(X0, P0, Np)
        
    def predict(self, dt):
        self.X = process_model_fly(self.X, dt, 1.5*self.Q, self.v, self.omega)
    
    def resample(self):
        idxs = np.random.choice(a = self.Np,size = self.Np, p = self.W)
        self.X = self.X[idxs]
        self.W = np.ones(self.Np)/self.Np
        
    def update(self, z):
        f = np.ones_like(self.W)
        for ii in range(len(z)):
            f = f * likelihood_model_fly(self.X, self.pose[ii], [z[ii]], [self.R[ii]])
        self.W = self.W * f
        if np.sum(self.W) == 0:
            self.W = np.ones(self.Np)/self.Np
        else:
            self.W = self.W/np.sum(self.W)
        
        Neff = 1/(np.sum(np.power(self.W, 2)))
        if Neff < (2/3)*self.Np:
            if self.verbos:
                print("Performing resample...")
            self.resample()
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
        self.X = process_model_sin(self.X, dt, self.Q)
    
    def update(self, z):
        
        self.W = self.W * likelihood_model_sin(self.X, z, self.R)
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
            
        