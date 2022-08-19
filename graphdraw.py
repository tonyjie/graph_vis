# python graphdraw.py figures/qh882/qh882.mat
import scipy.io as io
import sys
import argparse
import os

def compute_stress(positions, d):
    '''
    Compute stress in a redundant way -- consistent with `full_graph_sgd.py`, compute TWICE. 
    '''
    stress = 0
    n = d.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pq = positions[i] - positions[j]
            mag = np.linalg.norm(pq)
            stress += (1/d[i,j]**2) * (d[i,j]-mag)**2
    return stress



parser = argparse.ArgumentParser("Graph Drawing SGD Algorithm")
parser.add_argument("input_file", type=str, help="input graph name")
args = parser.parse_args()

# load the data from the SuiteSparse Matrix Collection format
#   https://www.cise.ufl.edu/research/sparse/matrices/
# mat_data = io.loadmat(graph_name + '.mat')
mat_data = io.loadmat(args.input_file)
graph = mat_data['Problem']['A'][0][0]


import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse.csgraph as csgraph

# find the shortest paths using dijkstra's
# note: if any path length is infinite i.e. we have disconnected subgraphs,
#       then this cell will work but the rest of the script won't.
d = csgraph.shortest_path(graph, directed=False, unweighted=True)
print(f"d.shape: {d.shape}") # (882, 882)
# print(f"d: {d}")



# show the shortest paths in a heat map.
# If any squares are white, then infinite paths exist and the algorithm will fail.
plt.imshow(d)
n = d.shape[0]

# initialise an array of indices to randomise relaxation order
constraints = []

# only use index pairs for i<j
w_min = float('inf')
w_max = 0
for i in range(n):
    for j in range(i):
        # w is w_ij from the MDS stress equation
        w = 1/d[i,j]**2
        w_min = min(w, w_min)
        w_max = max(w, w_max)
        constraints.append((i,j,w))


# determine the annealing schedule
num_iter = 15
epsilon = 0.1

eta_max = 1/w_min
eta_min = epsilon/w_max

lambd = np.log(eta_min / eta_max) / (num_iter - 1)
eta = lambda t: eta_max*np.exp(lambd*t)

# set up the schedule as an exponential decay
schedule = []
for i in range(num_iter):
    schedule.append(eta(i))
    
plt.plot(schedule)
# plt.show()


# Iteration Begin
import random
# initialise an array of 2D positions
positions = np.random.rand(n, 2)

import datetime
start = datetime.datetime.now()

for c_idx, c in enumerate(schedule):
    # shuffle the relaxation order
    random.shuffle(constraints)
        
    for idx, (i,j,w) in enumerate(constraints):
        wc = w*c
        if (wc > 1): # if not, there would be nan (super large gradient)
            wc = 1
        
        pq = positions[i] - positions[j]
        # mag is |p-q|
        mag = np.linalg.norm(pq)
        # r is the minimum distance each vertex has to move to satisfy the constraint
        r = (d[i,j] - mag) / 2
        m = wc * r * pq/mag

        # if (idx % 10000 == 0):
        #     print(f"idx[{idx}]: m = {m}")

        positions[i] += m
        positions[j] -= m

    stress = compute_stress(positions, d)
    print(f'Iteration {c_idx}: Stress = {stress:.1f}')

end = datetime.datetime.now()
print(end - start)



# Draw SVG Graph
from matplotlib import collections as mc

plt.axis('equal')
ax = plt.axes()
ax.set_xlim(min(positions[:,0])-1, max(positions[:,0])+1)
ax.set_ylim(min(positions[:,1])-1, max(positions[:,1])+1)

lines = []
for i,j in zip(*graph.nonzero()):
    if i > j:
        lines.append([positions[i], positions[j]])

lc = mc.LineCollection(lines, linewidths=1, colors='k', alpha=.5)
ax.add_collection(lc)

plt.savefig(args.input_file.replace('.mat', '.png'), format='svg', dpi=1000)

stress = 0
# calculate the stress
# for i in range(n):
#     for j in range(i):
#         pq = positions[i] - positions[j]
#         mag = np.linalg.norm(pq)
        
#         stress += (1/d[i,j]**2) * (d[i,j]-mag)**2

# calculate the stress in a redundant way -- consistent with `full_graph_sgd.py`
stress = compute_stress(positions, d)
print(f'stress = {stress:.1f}')
# print(f'stress = {stress:.2e}')
