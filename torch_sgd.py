import torch
import scipy.sparse.csgraph as csgraph
import scipy.io as io
import numpy as np
import random
import matplotlib.pyplot as plt

from matplotlib import collections as mc
from torch import nn
from itertools import combinations
import sys

from torch.utils.data import TensorDataset, DataLoader


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, d):
        n = d.shape[0] # number of nodes

        real_dis = []
        i_arr = []
        j_arr = []
        w_arr = []
        # only use index pairs for i<j
        w_min = float('inf')
        w_max = 0
        for j in range(n):
            for i in range(j):
                # w is w_ij from the MDS stress equation
                w = 1/d[i,j]**2
                w_min = min(w, w_min)
                w_max = max(w, w_max)
                i_arr.append(i)
                j_arr.append(j)
                w_arr.append(w)
                real_dis.append(d[i,j])

        self.i_arr = i_arr
        self.j_arr = j_arr
        self.w_arr = w_arr
        self.real_dis = real_dis

        self.w_min = w_min
        self.w_max = w_max

    def __getitem__(self, index): # return i, j, w, dis[i, j]
        return self.i_arr[index], self.j_arr[index], self.w_arr[index], self.real_dis[index]

    def __len__(self):
        return len(self.real_dis)



def draw_svg(positions, graph, output_name):
    # Draw SVG Graph
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

    plt.savefig(output_name + '.svg', format='svg', dpi=1000)




def compute_stress(vis_pos, real_dis):
    # compute the stress function
    n = vis_pos.shape[0]
    stress = 0
    for i in range(n):
        for j in range(i):
            w = 1 / (real_dis[i,j]**2)
            stress += w * (np.linalg.norm(vis_pos[i] - vis_pos[j]) - real_dis[i, j]) ** 2
    return stress

def q(t1, t2, dis):
    # one term in the stress function
    w = 1 / (dis**2)
    return w * ((torch.norm(t1 - t2) - dis) ** 2)


def main():

    # ========== Load Graph, Get the Ground Truth (shortest distance here) ============
    graph_name = 'qh882'
    mat_data = io.loadmat(graph_name + '.mat')
    graph = mat_data['Problem']['A'][0][0]

    # find the shortest paths using dijkstra's
    # note: if any path length is infinite i.e. we have disconnected subgraphs,
    #       then this cell will work but the rest of the script won't.
    d = csgraph.shortest_path(graph, directed=False, unweighted=True)
    # print(f"d.shape: {d.shape}") # (882, 882)
    # print(f"d: {d}")
    n = d.shape[0] # number of nodes

    # ========== Get Constraints and Input Data ============
    dataset = MyDataset(d)
    my_dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # Batch size need to be FIXED to 1

    w_min = dataset.w_min
    w_max = dataset.w_max

    # ========== Determine the annealing schedule (Step Size) ===========
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

    # print(f"schedule: {schedule}")

    # ========== Initialize the Positions ============
    positions = np.random.rand(n, 2)
    x = torch.tensor(positions, requires_grad=True)
    # print(f"x.shape: {x.shape}") # (882, 2)
    
    stress = compute_stress(positions, d)
    print(f"initial stress: {stress}")


    for idx_c, c in enumerate(schedule):

        for i, j, w, dis in my_dataloader: # batch size = 1
            wc = w * c
            if (wc > 1):
                wc = 1
            
            stress = q(x[i], x[j], dis)
            stress.backward()
            x.data.sub_(wc / (4*w) * x.grad.data) # really need this "/4"
            x.grad.data.zero_() 
        
        print(f"Iteration {idx_c} Done...")

    # Draw the Visualization Graph
    x_np = x.detach().numpy()

    draw_svg(x_np, graph, "sgd_draw")
    
    # Compute the Total Stress
    stress_total = compute_stress(x_np, d)
    print(f"stress_total: {stress_total}")


if __name__ == "__main__":
    main()