#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from particlefilter import particle_filter


class simulation():
    def __init__(self, Q, R, dt, Ns, P0, X0):
        self.X = np.zeros(Ns)
        self.X[0] = np.random.normal(X0, P0)
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
        self.X[0] = np.random.normal(self.X0, self.P0)
        
    def step(self):
        if self.t < self.Ns - 1:
            self.X[self.t+1] = np.random.normal(np.sin(self.dt*self.X[self.t]), self.Q)
            self.z = np.random.multivariate_normal(np.ones(self.Nz)*self.X[self.t+1], self.R)
            self.t +=1
        return self.X[self.t], self.z
    
    def visualize(self):
        plt.plot(self.X)
        plt.show()
        
        
if __name__ == '__main__':
    sim = simulation(Q = 0.5, R = np.array([1,2]), dt = 0.1, Ns = 100, Nz = 2, P0 = 0.5, X0 = 0)
    pf = particle_filter(X0 = sim.X0, P0 = sim.P0, R = sim.R[0,0],Q=sim.Q , Np = 100)
    mse = np.zeros(sim.Ns)
    x_hat = np.zeros(sim.Ns)
    for ii in range(sim.Ns-1):
        x,z=sim.step()
        pf.predict(dt = 0.1)
        pf.update(z = z[0])
        x_hat[ii+1] = pf.estimate()
        mse[ii+1] = (X_hat - x)**2
        print([x,z])
    plt.plot(sim.X)
    plt.plot(x_hat)
    plt.show()
