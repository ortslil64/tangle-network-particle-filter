#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tnpf.particlefilter import particle_filter_sin, particle_filter_fly
from tnpf.models import process_model_sin, mesurment_model_sin, process_model_fly, mesurment_model_fly

class simulation_sin():
    def __init__(self, Q, R, dt, Ns, P0, X0):
        self.X = np.zeros(Ns)
        self.X[0] = np.random.normal(X0, np.sqrt(P0))
        self.Q = Q
        self.R = np.diag(R)
        self.dt = dt
        self.Ns = Ns
        self.Nz = len(R)
        self.X0 = X0
        self.P0 = P0
        self.t = 0
        
    def reset(self):
        self.t = 0
        self.X = np.zeros(self.Ns)
        self.X[0] = np.random.normal(self.X0, np.sqrt(self.P0))
        
    def step(self):
        if self.t < self.Ns - 1:
            self.X[self.t+1] = process_model_sin(self.X[self.t], self.dt, self.Q)
            self.z = mesurment_model_sin(self.X[self.t+1], self.R)
            self.t +=1
        return self.X[self.t], self.z
    
    def visualize(self):
        plt.plot(self.X)
        plt.show()
        
class simulation_fly():
    def __init__(self, Q, R, dt, Ns, P0, X0, v, omega, poses):
        self.X = np.zeros((Ns,3))
        self.X[0,:] = np.random.multivariate_normal(X0, P0)
        self.Q = Q
        self.R = R
        self.dt = dt
        self.Ns = Ns
        self.Nz = len(R)
        self.X0 = X0
        self.P0 = P0
        self.t = 0
        self.v = v
        self.omega = omega
        self.poses = poses
        
    def reset(self):
        self.t = 0
        self.X = np.zeros((self.Ns,3))
        self.X[0,:] = np.random.multivariate_normal(self.X0, self.P0)
        
    def step(self):
        if self.t < self.Ns - 1:
            self.X[self.t+1,:] = process_model_fly(self.X[self.t,:].reshape(1,-1), self.dt, self.Q, self.v , self.omega)
            self.z = mesurment_model_fly(self.X[self.t+1], self.poses, self.R)
            self.t +=1
        return self.X[self.t], self.z
    

    
        
        
if __name__ == '__main__':
    sim_sin = simulation_sin(Q = 0.5, R = np.array([1,2]), dt = 0.1, Ns = 100,P0 = 0.5, X0 = 0)
    
    sim_fly = simulation_fly(Q = np.diag([0.1,0.1,0.01]),
                             R = [np.diag([0.1,0.01])]*3,
                             dt = 0.1,
                             Ns = 100,
                             P0 = np.diag([0.1,0.1,0.01]),
                             X0 = np.array([0,0,0]),
                             v = 1,
                             omega = 0.5,
                             poses = [np.array([0.0,0.0]), np.array([1.0,0.0]), np.array([0.0,1.0])])
    
    
    pf = particle_filter_fly(X0 = sim_fly.X0, P0 = sim_fly.P0, R = sim_fly.R[0], Q=sim_fly.Q , pose = sim_fly.poses[0], v = sim_fly.v, omega = sim_fly.omega , Np = 1000)
    x_hat = np.empty_like(sim_fly.X)
    for ii in range(sim_fly.Ns-1):
        x,z=sim_fly.step()
        pf.predict(dt = sim_fly.dt)
        pf.update(z = z[0])
        x_hat[ii,:] = pf.estimate()
    plt.scatter(sim_fly.X[:,0], sim_fly.X[:,1], c = 'black')   
    plt.scatter(x_hat[:,0], x_hat[:,1], c = 'blue')
    plt.show()
    
