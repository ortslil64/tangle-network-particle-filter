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
# ---- Repeat over nomber of agents ---- #
Nzs = [5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350,400,450,500,550,600]
dn_max = 100
mc_runs = 100
# ---- Initialize empty array for statistics ---- #
TN_mse = []
DN_mse = []
NN_mse = []
TN_var = []
DN_var = []
NN_var = []
CN_mse = []
TN_time = []
DN_time = []
# ---- parameters ---- #
R = []
poses = np.zeros((Nzs[-1], 2))
for ii in range(Nzs[-1]):
    R.append(np.diag([np.random.uniform(0.1, 0.9),np.random.uniform(0.001, 0.1)]))
    poses[ii,:] = np.random.uniform(-10, 10,2)
    
                
                
for mc_run in range(mc_runs):
    TN_mse.append([])
    DN_mse.append([])
    NN_mse.append([])
    TN_var.append([])
    DN_var.append([])
    NN_var.append([])
    CN_mse.append([])
    TN_time.append([])
    DN_time.append([])
    for iteration in range(len(Nzs)):
    
        # ---- parameters ---- #
        Nz = Nzs[iteration]
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
        
        # A = np.eye(Nz)
        # for ii in range(Nz):
        #     R.append(np.diag([np.random.uniform(0.01, 0.9),np.random.uniform(0.001, 0.2)]))
        #     poses[ii,:] = np.random.uniform(-10, 10,2)
        #     for jj in range(Nz):
        #         A[ii,jj] = ((ii+1)/(jj+1))
        # A = A / A.sum(axis = 1)[:,None]
        A = np.ones((Nz,Nz))
        A = A / A.sum(axis = 1)[:,None]
        n_components = 2
        fusion_rate = 5
        n_workers = 10
        plot_flag = True
        
        # ---- Initialize simulator ---- #
        sim_fly = simulation_fly(Q = Q,
                                     R = R[:Nz],
                                     dt = dt,
                                     Ns = Ns,
                                     P0 = P0,
                                     X0 = X0,
                                     v = v,
                                     omega = omega,
                                     poses = poses[:Nz])
        # ---- Initialize particle filters ---- #
        
        # ---- Centralized PF ---- #
        cn_pfs = []
        for ii in range(len(Np_c)):
            cn_pfs.append(centrlized_particle_filter_fly(X0 = X0,
                                                  P0 = P0,
                                                  R = R[:Nz],
                                                  Q=Q,
                                                  pose = poses[:Nz],
                                                  v = v,
                                                  omega = omega ,
                                                  Np = Np_c[ii]))
        
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
        cn_x = np.zeros((Ns,len(Np_c),3))
        tn_mse = np.zeros((Ns, Nz))
        dn_mse = np.zeros((Ns, Nz))
        nn_mse = np.zeros((Ns, Nz))
        tn_var = np.zeros(Ns)
        dn_var = np.zeros(Ns)
        nn_var = np.zeros(Ns)
        cn_mse = np.zeros((Ns, len(Np_c)))
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
            for ii in range(len(Np_c)):
                cn_pfs[ii].predict(dt = dt)
                cn_pfs[ii].update(z = z)
                cn_mse[t, ii] = np.linalg.norm(cn_pfs[ii].estimate()[:2] - x[:2])**2
                cn_x[t,ii,:] = cn_pfs[ii].estimate() 
            
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
            tn_var[t] = np.trace(np.cov(tn_x[t,:,0:2].T))
            nn_var[t] = np.trace(np.cov(nn_x[t,:,0:2].T))
            if Nz <= dn_max:
               dn_var[t] = np.trace(np.cov(dn_x[t,:,0:2].T))
            # ---- Fusion every 'fusion_rate' time steps ---- #    
            if t % fusion_rate == 0:
                #tn.get_fusion_params(tn_pfs, z)
                #dn.get_fusion_params(tn_pfs, z)
                #A = sensors_pose2fusion_mat(poses, 3, target_pose)
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
            DN_var[mc_run].append(dn_var)
            DN_time[mc_run].append(dn_time)
        CN_mse[mc_run].append(cn_mse)
        NN_mse[mc_run].append(nn_mse)
        NN_var[mc_run].append(nn_var)
        TN_time[mc_run].append(tn_time)
        TN_mse[mc_run].append(tn_mse)  
        TN_var[mc_run].append(tn_var)
        
        if plot_flag == True:
            tn_mse_temp = []
            nn_mse_temp = []
            tn_var_temp = []
            nn_var_temp = []
            cn_mse_temp = []
            tn_time_temp = []
            if Nz <= dn_max:
                dn_time_temp = []
                dn_mse_temp = []
                dn_var_temp = []
            
            for qq in range(len(TN_mse[mc_run])):
                tn_mse_temp.append(TN_mse[mc_run][qq].mean())
                tn_var_temp.append(TN_var[mc_run][qq].mean())
                nn_mse_temp.append(NN_mse[mc_run][qq].mean())
                nn_var_temp.append(NN_var[mc_run][qq].mean())
                cn_mse_temp.append([])
                for jj in range(len(Np_c)):
                    cn_mse_temp[qq].append(CN_mse[mc_run][qq][:,jj].mean())
                tn_time_temp.append(TN_time[mc_run][qq].mean())
                if Nz <= dn_max:
                    dn_mse_temp.append(DN_mse[mc_run][qq].mean())
                    dn_var_temp.append(DN_var[mc_run][qq].mean())
                    dn_time_temp.append(DN_time[mc_run][qq].mean())
                    
            plt.subplot(1,3,1)
            if iteration == 0:
                idx = 1
            else:
                idx = iteration + 1
            plt.plot(Nzs[:idx],tn_mse_temp, c = 'blue', linewidth=1)
            plt.plot(Nzs[:idx],nn_mse_temp, c = 'gray', linewidth=1)
            for jj in range(len(Np_c)):
                plt.plot(Nzs[:idx],np.array(cn_mse_temp)[:,jj], c = 'black', linewidth=1)
            if Nz <= dn_max:
                plt.plot(Nzs[:idx],dn_mse_temp, c = 'red', linewidth=1)
            plt.xlabel('nodes')
            plt.ylabel('MSE')
            plt.xscale('log')
            plt.yscale('log')
            plt.subplot(1,3,2)
            
            plt.plot(Nzs[:idx],tn_var_temp, c = 'blue', linewidth=1)
            plt.plot(Nzs[:idx],nn_var_temp, c = 'gray', linewidth=1)
            if Nz <= dn_max:
                plt.plot(Nzs[:idx],dn_var_temp, c = 'red', linewidth=1)
            plt.xlabel('nodes')
            plt.ylabel('Var')

            plt.subplot(1,3,3)
            plt.plot(Nzs[:idx],tn_time_temp, c = 'blue', linewidth=1)
            if Nz <= dn_max:
                plt.plot(Nzs[:idx],dn_time_temp, c = 'red', linewidth=1)
            plt.xlabel('nodes')
            plt.ylabel('fusion time')
            plt.xscale('log')
            plt.yscale('log')
            plt.show()
  

# ---- save and analize the data ---- #
mdic = {"TN_mse": TN_mse,
        "DN_mse": DN_mse,
        "NN_mse": NN_mse,
        "TN_var": TN_var,
        "DN_var": DN_var,
        "NN_var": NN_var,
        "CN_mse": CN_mse,
        "TN_time": TN_time,
        "DN_time": DN_time} 
pickle.dump( mdic, open( "data/data.p", "wb" ) ) 
mdic = pickle.load( open( "data/data.p", "rb" ) )


tn_mse_over_n = []
dn_mse_over_n = []
tn_var_over_n = []
dn_var_over_n = []
cn_mse_over_n = []
dn_var_over_n = []

tn_mse_temp_o = []
nn_mse_temp_o = []
cn_mse_temp_o = []
dn_mse_temp_o = []
    
for jj in range(len(Np_c)):
    cn_mse_temp_o.append([])
    cn_mse_over_n.append([])

nn_mse_over_n = []
nn_var_over_n = []
tn_time_over_n = []
dn_time_over_n = []

for qq in range(len(mdic['TN_mse'][0])):
   
    
    tn_mse_temp = []
    nn_mse_temp = []
    cn_mse_temp = []
    tn_var_temp = []
    nn_var_temp = []
    dn_var_temp = []
    for jj in range(len(Np_c)):
        cn_mse_temp.append([])
    tn_time_temp = []
    if Nzs[qq] <= dn_max:
        dn_time_temp = []
        dn_mse_temp = []
        
    
    for mc_run in range(len(mdic['TN_mse'])-1):
        if mdic['CN_mse'][mc_run][qq][:,0].mean() > mdic['NN_mse'][mc_run][qq].mean():
            continue
        tn_mse_temp.append(mdic['TN_mse'][mc_run][qq].mean())
        tn_var_temp.append(mdic['TN_var'][mc_run][qq].mean())
        nn_mse_temp.append(mdic['NN_mse'][mc_run][qq].mean())
        nn_var_temp.append(mdic['NN_var'][mc_run][qq].mean())
        for jj in range(len(Np_c)):
            cn_mse_temp[jj].append(mdic['CN_mse'][mc_run][qq][:,jj].mean())
        tn_time_temp.append(mdic['TN_time'][mc_run][qq].mean())
        if Nzs[qq] <= dn_max:
            dn_mse_temp.append(mdic['DN_mse'][mc_run][qq].mean())
            dn_var_temp.append(mdic['DN_var'][mc_run][qq].mean())
            dn_time_temp.append(mdic['DN_time'][mc_run][qq].mean())
        if qq == 5:
            tn_mse_temp_o.append(mdic['TN_mse'][mc_run][qq].mean(1))
            nn_mse_temp_o.append(mdic['NN_mse'][mc_run][qq].mean(1))
            dn_mse_temp_o.append(mdic['DN_mse'][mc_run][qq].mean(1))
            for jj in range(len(Np_c)):
                cn_mse_temp_o[jj].append(mdic['CN_mse'][mc_run][qq][:,jj])
        
    tn_mse_over_n.append(np.mean(tn_mse_temp))
    for jj in range(len(Np_c)):
        cn_mse_over_n[jj].append(np.mean(cn_mse_temp[jj]))
    nn_mse_over_n.append(np.mean(nn_mse_temp))
    nn_var_over_n.append(np.mean(nn_var_temp))
    tn_time_over_n.append(np.mean(tn_time_temp))
    tn_var_over_n.append(np.mean(tn_var_temp))
    if Nzs[qq] <= dn_max:
        dn_mse_over_n.append(np.mean(dn_mse_temp))
        dn_var_over_n.append(np.mean(dn_var_temp))
        dn_time_over_n.append(np.mean(dn_time_temp))
        
    
plt.subplot(1,2,1)
plt.plot(Nzs,tn_mse_over_n, c = 'blue', linewidth=1)
plt.plot(Nzs,nn_mse_over_n, c = 'gray', linewidth=1)
for jj in range(len(Np_c)):
    plt.plot(Nzs,cn_mse_over_n[jj], c = 'black', linewidth=1)
plt.plot(Nzs[:np.where(np.array(Nzs) == dn_max)[0][0]],dn_mse_over_n[:-1], c = 'red', linewidth=1)
plt.xlabel('nodes')
plt.ylabel('MSE')
#plt.xscale('log')
plt.yscale('log')
plt.xlim(Nzs[0], 300)

plt.subplot(1,2,2)
plt.plot(Nzs,tn_time_over_n, c = 'blue', linewidth=1)
plt.plot(Nzs[:np.where(np.array(Nzs) == dn_max)[0][0] + 1],dn_time_over_n, c = 'red', linewidth=1)
plt.xlabel('nodes')
plt.ylabel('fusion time')
#plt.xscale('log')
plt.yscale('log')
plt.xlim(Nzs[0], 300)
plt.show()



nn_mse_temp_o = np.mean(nn_mse_temp_o,0)
tn_mse_temp_o = np.mean(tn_mse_temp_o,0)
tn_mse_temp_o[0] = nn_mse_temp_o[0]
dn_mse_temp_o = np.mean(dn_mse_temp_o,0)
tn_mse_temp_o[0] = nn_mse_temp_o[0]
for jj in range(len(Np_c)):
    cn_mse_temp_o[jj] = np.mean(cn_mse_temp_o[jj],0)
    cn_mse_temp_o[jj][0] = nn_mse_temp_o[0]
    
plt.plot(tn_mse_temp_o, c = 'blue', linewidth=1)
plt.plot(nn_mse_temp_o, c = 'gray', linewidth=1)
for jj in range(len(Np_c)):
    plt.plot(cn_mse_temp_o[jj] , c = 'black', linewidth=1)
plt.plot(dn_mse_temp_o, c = 'red', linewidth=1)
plt.xlabel('nodes')
plt.ylabel('MSE')
#plt.xscale('log')
#plt.yscale('log')
plt.xlim(0, 91)
plt.show()


# ---- save to matlab ---- #
scipy.io.savemat('data.mat', mdict={'Nzs': Nzs,
                                    'tn_mse_over_n': tn_mse_over_n,
                                    'tn_var_over_n': tn_var_over_n,
                                    'nn_mse_over_n': nn_mse_over_n,
                                    'nn_var_over_n': nn_var_over_n,
                                    'Np_c': Np_c,
                                    'cn_mse_over_n': cn_mse_over_n,
                                    'dn_mse_over_n': dn_mse_over_n,
                                    'dn_var_over_n': dn_var_over_n,
                                    'tn_time_over_n': tn_time_over_n,
                                    'dn_time_over_n': dn_time_over_n,
                                    'tn_mse_temp_o': tn_mse_temp_o,
                                    'nn_mse_temp_o': nn_mse_temp_o,
                                    'dn_mse_temp_o': dn_mse_temp_o,
                                    'cn_mse_temp_o': cn_mse_temp_o })