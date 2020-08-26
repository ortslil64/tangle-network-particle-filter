#!/usr/bin/env python3
import numpy as np
from  simulator import simulation
from particlefilter import particle_filter
from tanglenetwork import tangle_network


# ---- parameters ---- #
Nz = 10
Q = 0.5
R = np.ones(Nz)
dt = 0.1 
Ns = 100
P0 = 0.5
X0 = 0
Np = 100
sigma = 0.01
# ---- Initialize simulator ---- #
sim = simulation(Q, R, dt, Ns, P0, X0)

# ---- Initialize particle filters ---- #
pfs = []
for ii in range(sim.Nz):
    pfs.append(particle_filter(X0 = sim.X0, P0 = sim.P0, R = R[ii], Q=sim.Q , Np = Np))

# ---- Initialize tangle network ---- #
tn = tangle_network(Na = sim.Nz, sigma = sigma, A = np.eye(Nz))

# ---- Initialize empty array for statistics ---- #
mse = np.zeros((sim.Ns, sim.Nz))


# ---- Iterate over time ---- #
for t in range(sim.Ns-1):
    x,z=sim.step()
    for ii in range(sim.Nz):
        pfs[ii].predict(dt = 0.1)
        pfs[ii].update(z = z[0])
        mse[t,ii] = (pfs[ii].estimate() - x)**2
    tn.get_fusion_params(pfs, z)
    pfs, calc_time = tn.fuse_particle_filters(pfs)

print("MSE: "+str(mse.mean()))        
        

