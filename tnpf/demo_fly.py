#!/usr/bin/env python3
import numpy as np
from  simulator import simulation_fly
from particlefilter import particle_filter_fly
from tanglenetwork import tangle_network
import matplotlib.pyplot as plt

# ---- parameters ---- #
Nz = 10
Q = np.diag([0.1,0.1,0.01])
R = [np.diag([0.2,0.03])]*Nz
dt = 0.1 
Ns = 100
P0 = np.diag([0.1,0.1,0.01])
v = 1
X0 = np.array([0,0,0])
Np = 1000
Np_c = 5000
sigma = 0.01
omega = 0.5
poses = np.zeros((Nz, 2))
A = np.eye(Nz)
for ii in range(Nz):
    poses[ii,:] = np.array([ii*10, ii*10])
    for jj in range(Nz):
        A[ii,jj] = ((ii+1)/(jj+1))**1.7
A = A / A.sum(axis = 1)[:,None]
n_components = 10
fusion_rate = 3
n_workers = None

# ---- Initialize simulator ---- #
sim_fly = simulation_fly(Q = Q,
                             R = R,
                             dt = dt,
                             Ns = Ns,
                             P0 = P0,
                             X0 = X0,
                             v = v,
                             omega = omega,
                             poses = poses)
# ---- Initialize particle filters ---- #
tn_pfs = []
for ii in range(Nz):
    tn_pfs.append(particle_filter_fly(X0 = X0,
                                      P0 = P0,
                                      R = R[ii],
                                      Q=Q,
                                      pose = poses[ii],
                                      v = v,
                                      omega = omega ,
                                      Np = Np))


pfs = []
for ii in range(Nz):
    pfs.append(particle_filter_fly(X0 = X0,
                                      P0 = P0,
                                      R = R[ii],
                                      Q=Q,
                                      pose = poses[ii],
                                      v = v,
                                      omega = omega ,
                                      Np = Np))
    
# ---- Initialize tangle network ---- #
tn = tangle_network(Na = Nz, sigma = sigma, A = A)

# ---- Initialize distributed particle filters network network ---- #


# ---- Initialize empty array for statistics ---- #
tn_x = np.zeros((Ns, Nz,3))
nn_x = np.zeros((Ns, Nz, 3))
dn_x = np.zeros((Ns, Nz))
cn_x = np.zeros(Ns)
tn_mse = np.zeros((Ns, Nz))
dn_mse = np.zeros((Ns, Nz))
nn_mse = np.zeros((Ns, Nz))
cn_mse = np.zeros(Ns)

# ---- Iterate over time ---- #
for t in range(Ns-1):
    x,z=sim_fly.step()
    
    for ii in range(Nz):
        tn_pfs[ii].predict(dt = dt)
        pfs[ii].predict(dt = dt)
        tn_pfs[ii].update(z = z[ii])
        pfs[ii].update(z = z[ii])
        tn_mse[t,ii] = np.linalg.norm(tn_pfs[ii].estimate()[:2] - x[:2])**2
        nn_mse[t,ii] = np.linalg.norm(pfs[ii].estimate()[:2] - x[:2])**2
        tn_x[t,ii,:] = tn_pfs[ii].estimate() 
        nn_x[t,ii,:] = pfs[ii].estimate()
    if t % fusion_rate == 0:
        #tn.get_fusion_params(tn_pfs, z)
        #dn.get_fusion_params(tn_pfs, z)
        tn_pfs, calc_time_tn = tn.fuse_particle_filters(tn_pfs, n_workers=n_workers)
    print("==========")
    print("Step:"+str(t)+" , TN time:"+str(calc_time_tn))
    print("MSE: TN:"+str(tn_mse.mean()))
    print("MSE: NN:"+str(nn_mse.mean()))
    
print("TN MSE: "+str(tn_mse.mean()))        
print("NN MSE: "+str(nn_mse.mean()))     


'''
1) Time compare
2) MSE
3) performance over node count
a) TN
b) centrelized
c) [18]
'''