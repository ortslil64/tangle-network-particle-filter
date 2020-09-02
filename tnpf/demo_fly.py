#!/usr/bin/env python3
import numpy as np
from  simulator import simulation_fly
from particlefilter import particle_filter_fly, centrlized_particle_filter_fly
from tanglenetwork import  log_tangle_network
from distributed_particle_filtering import DPF
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle

# ---- Repeat over nomber of agents ---- #
Nzs = [5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 1000, 1500, 2000]
dn_max = 100
mc_runs = 100
# ---- Initialize empty array for statistics ---- #
TN_mse = []
DN_mse = []
NN_mse = []
CN_mse = []
TN_time = []
DN_time = []
for mc_run in range(mc_runs):
    TN_mse.append([])
    DN_mse.append([])
    NN_mse.append([])
    CN_mse.append([])
    TN_time.append([])
    DN_time.append([])
    for iteration in range(len(Nzs)):
    
        # ---- parameters ---- #
        Nz = Nzs[iteration]
        Q = np.diag([0.1,0.1,0.01])
        R = [np.diag([0.2,0.03])]*Nz
        dt = 0.1 
        Ns = 100
        P0 = np.diag([0.1,0.1,0.01])
        v = 1
        X0 = np.array([0,0,0])
        Np = 100
        Np_c = 5000
        sigma = 0.01
        omega = 0.5
        poses = np.zeros((Nz, 2))
        A = np.eye(Nz)
        for ii in range(Nz):
            poses[ii,:] = np.random.uniform(-200, 200,2)
            for jj in range(Nz):
                A[ii,jj] = ((ii+1)/(jj+1))
        A = A / A.sum(axis = 1)[:,None]
        n_components = 3
        fusion_rate = 10
        n_workers = 16
        plot_flag = True
        
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
        
        # ---- Centralized PF ---- #
        cpf = centrlized_particle_filter_fly(X0 = X0,
                                              P0 = P0,
                                              R = R,
                                              Q=Q,
                                              pose = poses,
                                              v = v,
                                              omega = omega ,
                                              Np = Np_c)
        
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
        
        # --- without network ---- #
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
        tn = log_tangle_network(Na = Nz, sigma = sigma, A = A)
        
        # ---- Initialize distributed particle filters network network ---- #
        dn = DPF(Na = Nz, n_components = n_components, A = A)
        
        # ---- Initialize empty array for statistics ---- #
        tn_x = np.zeros((Ns, Nz,3))
        nn_x = np.zeros((Ns, Nz, 3))
        dn_x = np.zeros((Ns, Nz, 3))
        cn_x = np.zeros((Ns,3))
        tn_mse = np.zeros((Ns, Nz))
        dn_mse = np.zeros((Ns, Nz))
        nn_mse = np.zeros((Ns, Nz))
        cn_mse = np.zeros(Ns)
        tn_time = np.zeros(Ns)
        dn_time = np.zeros(Ns)
        
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
            
            # ---- Centralized PF ---- #
            cpf.predict(dt = dt)
            cpf.update(z = z)
            cn_mse[t] = np.linalg.norm(cpf.estimate()[:2] - x[:2])**2
            cn_x[t,:] = cpf.estimate() 
            
            # --- without network ---- #
            if n_workers is None:
                pool = mp.Pool(mp.cpu_count())
            else:
                pool = mp.Pool(n_workers)
            pfs = pool.map(pfs_worker, ((pfs, ii, z) for ii in range(Nz)))
            pool.close()
            pool.join()
            
            # --- tangle network ---- #
            if n_workers is None:
                pool = mp.Pool(mp.cpu_count())
            else:
                pool = mp.Pool(n_workers)
            tn_pfs = pool.map(pfs_worker, ((tn_pfs, ii, z) for ii in range(Nz)))
            pool.close()
            pool.join()
            
            # --- distributed network ---- #
            if Nz <= dn_max:
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
                # --- without network ---- #
                nn_mse[t,ii] = np.linalg.norm(pfs[ii].estimate()[:2] - x[:2])**2
                nn_x[t,ii,:] = pfs[ii].estimate()
                # --- distributed network ---- #
                if Nz <= dn_max:
                    dn_mse[t,ii] = np.linalg.norm(dn_pfs[ii].estimate()[:2] - x[:2])**2
                    dn_x[t,ii,:] = dn_pfs[ii].estimate() 
               
            # ---- Fusion every 'fusion_rate' time steps ---- #    
            if t % fusion_rate == 0:
                #tn.get_fusion_params(tn_pfs, z)
                #dn.get_fusion_params(tn_pfs, z)
                tn_pfs, calc_time_tn = tn.fuse_particle_filters(tn_pfs, n_workers=n_workers)
                if Nz <= dn_max:
                    dn_pfs, calc_time_dn = dn.fuse_particle_filters(dn_pfs, n_workers=n_workers)
            tn_time[t] = calc_time_tn  
            if Nz <= dn_max:
                dn_time[t] = calc_time_dn  
            print("==========")
            print("Nz: "+str(Nz) + "mc_run: "+str(mc_run))
            print("Step:"+str(t)+" , TN time:"+str(calc_time_tn))
            if Nz <= dn_max:
                print("Step:"+str(t)+" , DN time:"+str(calc_time_dn))
                print("MSE: DN:"+str(dn_mse.mean()))
            print("MSE: TN:"+str(tn_mse.mean()))
            print("MSE: NN:"+str(nn_mse.mean()))
            print("MSE: CN:"+str(cn_mse.mean()))
        if Nz <= dn_max:
            DN_mse[mc_run].append(dn_mse)
            DN_time[mc_run].append(dn_time)
        CN_mse[mc_run].append(cn_mse)
        NN_mse[mc_run].append(nn_mse)
        TN_time[mc_run].append(tn_time)
        TN_mse[mc_run].append(tn_mse)   
        
        if plot_flag == True:
            tn_mse_temp = []
            nn_mse_temp = []
            cn_mse_temp = []
            tn_time_temp = []
            if Nz <= dn_max:
                dn_time_temp = []
                dn_mse_temp = []
            
            for qq in range(len(TN_mse[mc_run])):
                tn_mse_temp.append(TN_mse[mc_run][qq].mean())
                nn_mse_temp.append(NN_mse[mc_run][qq].mean())
                cn_mse_temp.append(CN_mse[mc_run][qq].mean())
                tn_time_temp.append(TN_time[mc_run][qq].mean())
                if Nz <= dn_max:
                    dn_mse_temp.append(DN_mse[mc_run][qq].mean())
                    dn_time_temp.append(DN_time[mc_run][qq].mean())
                    
            plt.subplot(1,2,1)
            if iteration == 0:
                idx = 1
            else:
                idx = iteration + 1
            plt.plot(Nzs[:idx],tn_mse_temp, c = 'blue', linewidth=1)
            plt.plot(Nzs[:idx],nn_mse_temp, c = 'gray', linewidth=1)
            plt.plot(Nzs[:idx],cn_mse_temp, c = 'black', linewidth=1)
            if Nz <= dn_max:
                plt.plot(Nzs[:idx],dn_mse_temp, c = 'red', linewidth=1)
            plt.xlabel('t')
            plt.ylabel('MSE')
            plt.subplot(1,2,2)
            plt.plot(Nzs[:idx],tn_time_temp, c = 'blue', linewidth=1)
            if Nz <= dn_max:
                plt.plot(Nzs[:idx],dn_time_temp, c = 'red', linewidth=1)
            plt.xlabel('t')
            plt.ylabel('fusion time')
            plt.show()
  

mdic = {"TN_mse": TN_mse,
        "DN_mse": DN_mse,
        "NN_mse": NN_mse,
        "CN_mse": CN_mse,
        "TN_time": TN_time,
        "DN_time": DN_time} 
pickle.dump( mdic, open( "save.p", "wb" ) ) 


