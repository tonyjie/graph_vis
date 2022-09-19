# python full_graph_sgd.py figures/qh882/qh882.mat --steps=400 --log_interval=10 --cuda --draw --draw_interval=10

import torch
import scipy.sparse.csgraph as csgraph
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import collections as mc
from torch import nn
import sys
import os
import datetime
import argparse
import imageio


def draw(pos_changes, graph, output_dir, DRAW_INTERVAL):
    # remove all the .png file in output_dir
    os.system(f"rm {output_dir}/*.png")

    frames = list()
    for idx, positions in enumerate(pos_changes):
        # Draw Graph
        fig, ax = plt.subplots()
        # plt.axis('equal')
        # ax = plt.axes()
        ax.set_title(f"step {idx * DRAW_INTERVAL}")
        ax.set_xlim(min(positions[:,0])-1, max(positions[:,0])+1)
        ax.set_ylim(min(positions[:,1])-1, max(positions[:,1])+1)

        lines = []
        for i,j in zip(*graph.nonzero()):
            if i > j:
                lines.append([positions[i], positions[j]])

        lc = mc.LineCollection(lines, linewidths=1, colors='k', alpha=.5)
        ax.add_collection(lc)

        fig.savefig(f"{output_dir}/{idx * DRAW_INTERVAL}.png")
        frames.append(imageio.imread(f"{output_dir}/{idx * DRAW_INTERVAL}.png"))
    
    imageio.mimsave(f"{output_dir}/animation.gif", frames, duration=0.1)


class PlaceEngine(nn.Module):
    '''
    @brief Graph 2D Layout Engine. It contains the parameters ([X, Y] coordinates of all nodes) to be updated. 
    '''

    def __init__(self, num_nodes, LR):
        '''
        @brief initialization
        @param num_nodes: number of nodes in the graph
        @param LR: learning rate
        '''
        super().__init__()
        self.num_nodes = num_nodes
        pos = torch.empty(num_nodes, 2)
        self.pos = nn.Parameter(data=pos, requires_grad=True)
        nn.init.uniform_(self.pos) # can try other initialization methods: kaiming_uniform; xavier_uniform

        # self.dist = dist # [num_nodes, num_nodes]
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR) # Adam -> Loss = 3.75e4
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=LR, momentum=0.9, nesterov=True) # Nesterov momentum
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2) # SGD
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-2)
    
    def gradient_step(self, dist):

        self.optimizer.zero_grad()

        # backward
        stress = self.stress_fn(dist)
        # print(f"stress: {stress}")
        stress.backward()
        # print(f"gradient: {self.pos.grad}")
        self.optimizer.step()

        return stress

    def stress_fn(self, dist):
        '''
        @brief Compute the stress function
        '''
        
        copy1 = torch.reshape(self.pos, (1, self.num_nodes, 2))
        copy2 = torch.reshape(self.pos, (self.num_nodes, 1, 2))
        broadcasted1 = torch.broadcast_to(copy1, (self.num_nodes, self.num_nodes, 2))
        broadcasted2 = torch.broadcast_to(copy2, (self.num_nodes, self.num_nodes, 2))
        diff = broadcasted1 - broadcasted2
        pred_dist = torch.norm(diff, dim=2).reshape((self.num_nodes, self.num_nodes))
        mask = dist.ne(0)
        pred_dist = torch.where(mask, pred_dist, dist)
        stress_matrix = torch.where(mask, torch.square((pred_dist - dist) / dist), dist) # [num_nodes, num_nodes]. Actually, this involves with redundant computation. (The matrix is symmetric)
        # stress_matrix = torch.where(mask, torch.square(pred_dist - dist), dist) # try no weight
        stress = torch.sum(stress_matrix)
        
        return stress
        





def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"==== Device: {device}; Dataset: {args.input_file} ====")


    STEPS = args.steps
    LR = args.lr
    DRAW_INTERVAL = args.draw_interval
    LOG_INTERVAL = args.log_interval





    mat_data = io.loadmat(args.input_file)
    graph = mat_data['Problem']['A'][0][0]
    dist = csgraph.shortest_path(graph, directed=False, unweighted=True)
    print(f"dist.shape: {dist.shape}") # (5, 5)
    # print(f"dist: {dist}")
    

    num_nodes = dist.shape[0]

    # dis = np.random.randn(num_nodes, num_nodes)
    dist = torch.tensor(dist, dtype=torch.float32)
    dist = dist.to(device)

    mod = PlaceEngine(num_nodes, LR)
    mod = mod.to(device)
    # print(mod.pos)

    pos_changes = np.zeros((STEPS//DRAW_INTERVAL, num_nodes, 2), dtype=np.float32)

    stress_rec = np.zeros((STEPS//LOG_INTERVAL,), dtype=np.float32)
    # ====== Training ======
    start = datetime.datetime.now()
    for i in range(STEPS):
        stress = mod.gradient_step(dist)
        mod.scheduler.step() # learning rate scheduler

        if i % LOG_INTERVAL == 0:
            print(f"step {i}/{STEPS}: {stress:.1f}")
            stress_rec[i//LOG_INTERVAL] = stress.item()

        if args.draw and i % DRAW_INTERVAL == 0:
            pos = mod.pos.cpu().detach().numpy()
            pos_changes[i // DRAW_INTERVAL] = pos

    end = datetime.datetime.now()

    print(f"==== Training time: {end - start}; Step: {STEPS}; Device: {device} ====")

    # # === draw learning curve ===
    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(0, STEPS, STEPS//LOG_INTERVAL), stress_rec)
    # ax.set_xlabel('Step')
    # ax.set_ylabel('Stress')
    # ax.set_yscale('log')
    # plt.savefig(f"{os.path.dirname(args.input_file)}/learning_curve.png")


    if args.draw:
        draw(pos_changes, graph, os.path.dirname(args.input_file), DRAW_INTERVAL)

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Graph SGD Update Implementation for Graph Drawing using Pytorch")
    parser.add_argument("input_file", type=str, help="input graph name")    
    # parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--steps', type=int, default=100, help='number of steps')
    parser.add_argument('--lr', type=float, default=1, help='learning rate')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--draw', action='store_true', default=False, help='Draw Animation')
    parser.add_argument('--draw_interval', type=int, default=1, help='Draw Animation Interval (Steps)')

    args = parser.parse_args()
    print(args)
    main(args)