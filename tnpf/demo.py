#!/usr/bin/env python3
import numpy as np
from  simulator import simulation_sin
from particlefilter import particle_filter, centrlized_particle_filter
from tanglenetwork import tangle_network
from distributed_particle_filtering import DPF
import matplotlib.pyplot as plt

# ---- parameters ---- #
Nz = 10
Q = 0.7
R = np.random.uniform(1.1,1.11, size=Nz)
dt = 0.5 
Ns = 70
P0 = 0.1
X0 = 0
Np = 200
Np_c = 5000
sigma = 0.01
n_components = 10
fusion_rate = 10
n_workers = None
A = np.eye(Nz) + 0.1*np.ones((Nz,Nz))
A = A / A.sum(axis = 1)[:,None]
# ---- Initialize simulator ---- #
sim = simulation_sin(Q, R, dt, Ns, P0, X0)

# ---- Initialize particle filters ---- #
tn_pfs = []
for ii in range(sim.Nz):
    tn_pfs.append(particle_filter(X0 = sim.X0, P0 = sim.P0, R = R[ii], Q=sim.Q , Np = Np))

dn_pfs = []
for ii in range(sim.Nz):
    dn_pfs.append(particle_filter(X0 = sim.X0, P0 = sim.P0, R = R[ii], Q=sim.Q , Np = Np))
    
pfs = []
for ii in range(sim.Nz):
    pfs.append(particle_filter(X0 = sim.X0, P0 = sim.P0, R = R[ii], Q=sim.Q , Np = Np))
    
cpf = centrlized_particle_filter(X0 = sim.X0, P0 = sim.P0, R = R, Q=sim.Q , Np = Np_c)
# ---- Initialize tangle network ---- #
tn = tangle_network(Na = sim.Nz, sigma = sigma)

# ---- Initialize distributed particle filters network network ---- #
dn = DPF(Na = sim.Nz, n_components = n_components)


# ---- Initialize empty array for statistics ---- #
tn_x = np.zeros((sim.Ns, sim.Nz))
nn_x = np.zeros((sim.Ns, sim.Nz))
dn_x = np.zeros((sim.Ns, sim.Nz))
cn_x = np.zeros(sim.Ns)
tn_mse = np.zeros((sim.Ns, sim.Nz))
dn_mse = np.zeros((sim.Ns, sim.Nz))
nn_mse = np.zeros((sim.Ns, sim.Nz))
cn_mse = np.zeros(sim.Ns)

# ---- Iterate over time ---- #
for t in range(sim.Ns-1):
    x,z=sim.step()
    cpf.predict(dt = dt)
    cpf.update(z = z)
    cn_mse[t] = (cpf.estimate() - x)**2
    cn_x[t] = cpf.estimate()
    for ii in range(sim.Nz):
        tn_pfs[ii].predict(dt = dt)
        dn_pfs[ii].predict(dt = dt)
        pfs[ii].predict(dt = dt)
        tn_pfs[ii].update(z = z[ii])
        dn_pfs[ii].update(z = z[ii])
        pfs[ii].update(z = z[ii])
        tn_mse[t,ii] = (tn_pfs[ii].estimate() - x)**2
        nn_mse[t,ii] = (pfs[ii].estimate() - x)**2
        dn_mse[t,ii] = (dn_pfs[ii].estimate() - x)**2
        tn_x[t,ii] = tn_pfs[ii].estimate()
        nn_x[t,ii] = pfs[ii].estimate()
        dn_x[t,ii] = dn_pfs[ii].estimate()
    if t % fusion_rate == 0:
        tn.get_fusion_params(tn_pfs, z)
        #dn.get_fusion_params(tn_pfs, z)
        tn_pfs, calc_time_tn = tn.fuse_particle_filters(tn_pfs, n_workers=n_workers)
        dn_pfs, calc_time_dn = dn.fuse_particle_filters(dn_pfs, n_workers=n_workers)
    print("==========")
    print("Step:"+str(t)+" , TN time:"+str(calc_time_tn)+" , DN time:"+str(calc_time_dn))
    print("MSE: TN:"+str(tn_mse.mean()))
    print("MSE: DN:"+str(dn_mse.mean()))
    print("MSE: CN:"+str(cn_mse.mean()))
    print("MSE: NN:"+str(nn_mse.mean()))
    
print("TN MSE: "+str(tn_mse.mean()))        
print("CN MSE: "+str(cn_mse.mean()))    
print("NN MSE: "+str(nn_mse.mean()))     
print("DN MSE: "+str(dn_mse.mean()))   

for ii in range(sim.Nz):
    plt.plot(tn_x[:,ii], c = 'blue', linewidth=0.2)
    plt.plot(dn_x[:,ii], c = 'green', linewidth=0.2)
    plt.plot(nn_x[:,ii], c = 'gray', linewidth=0.2)
plt.plot(sim.X[1:], c = 'black', linewidth=1)     
plt.plot(cn_x, c = 'red', linewidth=0.5) 
plt.show()

'''
1) Time compare
2) MSE
3) performance over node count
a) TN
b) centrelized
c) [18]
'''