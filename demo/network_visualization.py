#!/usr/bin/env python3
import numpy as np
from  simulator import simulation_fly
import matplotlib.pyplot as plt
import networkx as nx
import scipy.io

# ---- parameters ---- #
Nzs = [5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500,600,700,800,900,1000]

R = []
poses = np.zeros((Nzs[-1], 2))
for ii in range(Nzs[-1]):
    R.append(np.diag([np.random.uniform(0.1, 0.9),np.random.uniform(0.001, 0.1)]))
    poses[ii,:] = np.random.uniform(-10, 10,2)
Q = np.diag([0.02,0.02,0.001])
dt = 0.1 
Ns = 300
P0 = np.diag([0.05,0.05,0.01])
v = 1
X0 = np.array([0,0,0])
Np = 100
sigma = 0.01
omega = 0.3
Nz = Nzs[5]
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
for ii in range(Ns):
    sim_fly.step()
# ---- Initialize graph ---- #
A = np.eye(Nz)
for ii in range(Nz):
    for jj in range(Nz):   
        if ii == jj:
            A[ii,jj] = 1/Nz
        else:             
            A[ii,jj] = 1/np.linalg.norm(poses[ii] - poses[jj])**2
A = A / A.sum(axis = 1)[:,None]
graph=nx.Graph()
for ii in range(Nz):
    graph.add_node(ii,pos = (poses[ii,0], poses[ii,1]))

for ii in range(Nz):
    for jj in range(Nz):
        if A[ii,jj] > 3/Nz and ii != jj:
            graph.add_edge(ii,jj, weight=40*A[ii,jj] )
edges = graph.edges()
weights = [graph[u][v]['weight'] for u,v in edges]
colors = [(graph[u][v]['weight']) for u,v in edges]
weights = colors
colors = 2*(np.array(colors) - min(colors))/(max(colors) - min(colors))
pos=nx.get_node_attributes(graph,'pos')
fig, ax = plt.subplots(figsize=(10,10))
nx.draw(graph,pos, node_size = 20,  edge_color='black' ,width = 1.2,  edge_cmap=plt.cm.Blues, ax=ax, node_color='blue')
limits=plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.plot(sim_fly.X[:,0], sim_fly.X[:,1], c='red', linewidth = 1.0)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.show()
# ---- plot connectivity ---- #
fig, (ax1,ax2) = plt.subplots(figsize=(5,10), nrows=2)
A = np.random.rand(Nz,Nz)
A = A / A.sum(axis = 1)[:,None]
pos = ax1.imshow(A, cmap=plt.cm.Greys)
fig.colorbar(pos, ax=ax1)

ax2.scatter(poses[:,0], poses[:,1])
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.show()

# ---- Save data ---- #
scipy.io.savemat('data/network_data.mat', mdict={'Nz': Nz,
                                    'node_pose': sim_fly.poses,
                                    'A': A,
                                    'target_path': sim_fly.X})

