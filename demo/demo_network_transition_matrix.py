#!/usr/bin/env python3
import numpy as np
from tnpf.simulator import simulation_fly
from tnpf.particlefilter import particle_filter_fly, centrlized_particle_filter_fly
from tnpf.tanglenetwork import  log_tangle_network, sensors_pose2fusion_mat
from tnpf.distributed_particle_filtering import DPF
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
import scipy.io
import scipy.linalg as la


def get_net_mat(lambda2, n):
    des = np.zeros(n)
    des[0] = 1.0
    des[1] = lambda2
    s = np.diag(des)
    v = np.random.rand(n, n)
    v = la.orth(v)
    v[:,0]=np.ones(n)
    v, _ = la.qr(v)
    semidef = v.dot(s.dot(v.T))
    semidef[semidef<0] = 0
    semidef = semidef / semidef.sum(axis = 1)[:,None]
    return semidef

# ---- Repeat over nomber of agents ---- #
mc_runs = 100
# ---- Initialize empty array for statistics ---- #
TN_mse = []
DN_mse = []
TN_var = []
DN_var = []

# ---- parameters ---- #
Nz = 40
Q = np.diag([0.02,0.02,0.003])
dt = 0.5 
Ns = 50
P0 = np.diag([0.5,0.5,0.01])
v = 1
X0 = np.array([0,0,0])
Np = 200
Np_c = [200, 500, 5000]
sigma = 0.01
omega = 0.4
n_components = 2
fusion_rate = 5
n_workers = 10
plot_flag = True
R = []
poses = np.zeros((Nz, 2))
lambda2 = np.arange(0.1,0.9,0.1)
for ii in range(Nz):
    R.append(np.diag([np.random.uniform(0.1, 0.9),np.random.uniform(0.001, 0.1)]))
    poses[ii,:] = np.random.uniform(-10, 10,2)
    
                
                
for mc_run in range(mc_runs):
    TN_mse.append([])
    DN_mse.append([])
    
    TN_var.append([])
    DN_var.append([])
    
    for lmb2_idx in range(len(lambda2)):
        # ---- network transition matrix parameters ---- #
        A = get_net_mat(lambda2[lmb2_idx], Nz)
        
        
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
        
       
        
        # --- tangle network ---- #
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
        
        # --- distributed network ---- #
        dn_pfs = []
        for ii in range(Nz):
            dn_pfs.append(particle_filter_fly(X0 = X0,
                                              P0 = P0,
                                              R = R[ii],
                                              Q=Q,
                                              pose = poses[ii],
                                              v = v,
                                              omega = omega ,
                                              Np = Np))
        
       
            
        # ---- Initialize tangle network ---- #
        tn = log_tangle_network(Na = Nz, sigma = sigma, A = A)
        
        # ---- Initialize distributed particle filters network network ---- #
        dn = DPF(Na = Nz, n_components = n_components, A = A)
        
        # ---- Initialize empty array for statistics ---- #
        tn_x = np.zeros((Ns, Nz,3))
        dn_x = np.zeros((Ns, Nz, 3))
        tn_mse = np.zeros((Ns, Nz))
        dn_mse = np.zeros((Ns, Nz))
        tn_var = np.zeros(Ns)
        dn_var = np.zeros(Ns)
        
        # ---- Parallel worker ---- #
        def pfs_worker(arg):
            pfs, ii, z = arg
            pfs[ii].predict(dt = dt)
            pfs[ii].update(z = z[ii])
            return pfs[ii]
        
            
            return pfs[ii]
        # ---- Iterate over time ---- #
        for t in range(Ns-1):
            x,z=sim_fly.step()
            
            # --- running all PFs parrallel ---- #
            
            # --- tangle network ---- #
            if n_workers is None:
                pool = mp.Pool(mp.cpu_count())
            else:
                pool = mp.Pool(n_workers)
            tn_pfs = pool.map(pfs_worker, ((tn_pfs, ii, z) for ii in range(Nz)))
            pool.close()
            pool.join()
            
            # --- distributed network ---- #
            if n_workers is None:
                pool = mp.Pool(mp.cpu_count())
            else:
                pool = mp.Pool(n_workers)
            dn_pfs = pool.map(pfs_worker, ((dn_pfs, ii, z) for ii in range(Nz)))
            pool.close()
            pool.join()
            
            # ---- estimating MSE for each pf in each network ---- #
            
            for ii in range(Nz):
                # --- tangle network ---- #
                tn_mse[t,ii] = np.linalg.norm(tn_pfs[ii].estimate()[:2] - x[:2])**2
                tn_x[t,ii,:] = tn_pfs[ii].estimate() 
                # --- distributed network ---- #
                dn_mse[t,ii] = np.linalg.norm(dn_pfs[ii].estimate()[:2] - x[:2])**2
                dn_x[t,ii,:] = dn_pfs[ii].estimate() 
            tn_var[t] = np.trace(np.cov(tn_x[t,:,0:2].T))
            dn_var[t] = np.trace(np.cov(dn_x[t,:,0:2].T))
            # ---- Fusion every 'fusion_rate' time steps ---- #    
            if t % fusion_rate == 0:

                tn_pfs, calc_time_tn = tn.fuse_particle_filters(tn_pfs, n_workers=n_workers)
                dn_pfs, calc_time_dn = dn.fuse_particle_filters(dn_pfs, n_workers=n_workers)

            print("==========")
            print("L2: "+str(lambda2[lmb2_idx]) + "mc_run: "+str(mc_run))
            print("DN: MSE:"+str(dn_mse.mean()) + "Var:"+str(dn_var.mean()))
            print("TN: MSE:"+str(tn_mse.mean()) + "Var:"+str(tn_var.mean()))

        DN_mse[mc_run].append(dn_mse)
        DN_var[mc_run].append(dn_var)
        TN_mse[mc_run].append(tn_mse)  
        TN_var[mc_run].append(tn_var)
        
        if plot_flag == True:
            plt.subplot(1,3,1)
            plt.plot(tn_mse.mean(1), c = 'blue', linewidth=1)
            plt.plot(dn_mse.mean(1), c = 'red', linewidth=1)
            plt.xlabel('nodes')
            plt.ylabel('MSE')
            plt.xscale('log')
            plt.yscale('log')
            plt.subplot(1,3,2)
            
            plt.plot(tn_var, c = 'blue', linewidth=1)
            plt.plot(dn_var, c = 'red', linewidth=1)
            plt.xlabel('nodes')
            plt.ylabel('Var')
            
            plt.subplot(1,3,3)
            
            plt.plot(np.array(TN_var[0]).mean(1), c = 'blue', linewidth=1)
            plt.plot(np.array(DN_var[0]).mean(1), c = 'red', linewidth=1)
            plt.xlabel('nodes')
            plt.ylabel('Var')
    
            plt.show()
  

# ---- save and analize the data ---- #
mdic = {"TN_mse": TN_mse,
        "DN_mse": DN_mse,
        "NN_mse": NN_mse,
        "TN_var": TN_var,
        "DN_var": DN_var,
        "NN_var": NN_var,
        "CN_mse": CN_mse} 



# ---- save to matlab ---- #
scipy.io.savemat('data.mat', mdict=mdic)