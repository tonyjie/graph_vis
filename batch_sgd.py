import odgi

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
import datetime
from torch.utils.data import TensorDataset, DataLoader
import argparse

class odgiDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        self.g = odgi.graph()
        self.g.load(file_name)

        print("Min node id: ", self.g.min_node_id())
        print("Max node id: ", self.g.max_node_id())
        assert self.g.min_node_id() == 1
        assert self.g.max_node_id() == self.g.get_node_count()

        self.path_names = []
        self.g.for_each_path_handle(lambda p: self.path_names.append(self.g.get_path_name(p)))

        self.sizes = []
        self.g.for_each_path_handle(lambda p: self.sizes.append(self.g.get_step_count(p)))

        self.ranges = []
        for idx, s in enumerate(self.sizes):
            combs = int(s * (s-1) / 2)
            if idx > 0:
                prev = self.ranges[idx-1]
                self.ranges.append(self.ranges[idx-1] + combs)
            else:
                self.ranges.append(combs)

        self.w_max = 1                          # w_max when nodes next to each other
        self.w_min = 1 / ((max(self.sizes)-1)**2)    # w_min when nodes on ends of longest path

        self.rnd_path_analysis = [0]*self.get_path_count()
        self.rnd_node_analysis = [0]*self.get_node_count()
        print(self.rnd_path_analysis)
        print(self.rnd_node_analysis)

        return

    def __getitem__(self, index): # return i, j, w, dis[i, j]
        # # get path and compute combination index in path
        # path = None
        # comb_idx = index
        # for (path_idx, r) in enumerate(self.ranges):
        #     if index < r:
        #         path = self.get_path(path_idx)
        #         if path_idx > 0:
        #             comb_idx = comb_idx - self.ranges[path_idx-1]
        #         print("path-index: ", path_idx)
        #         print("comb-index: ", comb_idx)
        #         break


        # random.seed(index)

        # choose random path
        path_idx = random.randrange(self.g.get_path_count())
        path = self.get_path(path_idx)

        # choose random step_a and step_b in path
        path_size = self.g.get_step_count(path)
        step_a_idx = random.randrange(path_size - 1)
        step_b_idx = random.randrange(step_a_idx + 1, path_size)
        assert step_a_idx < step_b_idx

        d = step_b_idx - step_a_idx
        w = 1/(d**2)

        # get id of step_a
        step = self.g.path_begin(path)
        for i in range(step_a_idx):
            assert self.g.has_next_step(step) == True
            step = self.g.get_next_step(step)

        id_step_a = self.g.get_id(self.g.get_handle_of_step(step))

        # get id of step_b
        for i in range(d):
            assert self.g.has_next_step(step) == True
            step = self.g.get_next_step(step)

        id_step_b = self.g.get_id(self.g.get_handle_of_step(step))

        self.rnd_path_analysis[path_idx] = self.rnd_path_analysis[path_idx] + 1
        self.rnd_node_analysis[id_step_a-1] = self.rnd_node_analysis[id_step_a-1] + 1
        self.rnd_node_analysis[id_step_b-1] = self.rnd_node_analysis[id_step_b-1] + 1

        return id_step_a, id_step_b, w, d

    def __len__(self):
        # return self.ranges[self.g.get_path_count()-1]
        return self.get_path_count() * self.get_node_count() * 10
        # return self.get_nbr_nodes() * (self.get_nbr_nodes()-1) * 2

    def get_path(self, idx):
        assert 0 <= idx < self.g.get_path_count()
        path_name = self.path_names[idx]
        return self.g.get_path_handle(path_name)

    def get_node_count(self):
        return self.g.get_node_count()

    def get_path_count(self):
        return self.g.get_path_count()

    def get_node_id_in_path(self, path, sidx):
        step = self.g.path_begin(path)
        for i in range(sidx):
            assert self.g.has_next_step(step) == True
            step = self.g.get_next_step(step)

        return self.g.get_id(self.g.get_handle_of_step(step))

    def get_step_cnt_in_path(self, path):
        return self.g.get_step_count(path)




def draw_svg(x, y, gdata, output_name):
    # Draw SVG Graph
    # plt.axis('equal')
    ax = plt.axes()
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))

    lines = []
    for pidx in range(gdata.get_path_count()): #NOTE: Draw all paths
        print("drawing path ", pidx)
        p = gdata.get_path(pidx)

        i = None
        j = None
        for sidx in range(gdata.get_step_cnt_in_path(p)):
            i = j
            j = gdata.get_node_id_in_path(p, sidx)
            if i != None:
                lines.append([[x[i-1],y[i-1]], [x[j-1],y[j-1]]])

    lc = mc.LineCollection(lines, linewidths=1, colors='k', alpha=.5)
    ax.add_collection(lc)

    plt.plot(x,y,marker='o', linestyle='', color='black')
    for i in range(gdata.get_node_count()):
        plt.text(x[i]+.01, y[i]+.01, i+1)

    plt.savefig('output/' + output_name + '.svg', format='svg', dpi=1000)




# def compute_stress(vis_pos, real_dis):
#     # compute the stress function
#     n = vis_pos.shape[0]
#     stress = 0
#     for i in range(n):
#         for j in range(i):
#             w = 1 / (real_dis[i,j]**2)
#             stress += w * (np.linalg.norm(vis_pos[i] - vis_pos[j]) - real_dis[i, j]) ** 2
#     return stress

def q(t1, t2, dis):
    # one term in the stress function
    w = 1 / (dis**2)
    return torch.mean(w * ((torch.norm((t1 - t2), dim=1) - dis) ** 2))


def main(args):
    # # ========== Load Graph, Get the Ground Truth (shortest distance here) ============
    # graph_name = 'qh882'
    # mat_data = io.loadmat(graph_name + '.mat')
    # graph = mat_data['Problem']['A'][0][0]

    # # find the shortest paths using dijkstra's
    # # note: if any path length is infinite i.e. we have disconnected subgraphs,
    # #       then this cell will work but the rest of the script won't.
    # d = csgraph.shortest_path(graph, directed=False, unweighted=True)
    # # print(f"d.shape: {d.shape}") # (882, 882)
    # # print(f"d: {d}")
    # n = d.shape[0] # number of nodes

    # ========== Get Constraints and Input Data ============
    dataset = odgiDataset(args.file)

    n = dataset.get_node_count()

    w_min = dataset.w_min
    w_max = dataset.w_max

    # ========== Determine the annealing schedule (Step Size). Parameter Setting. ===========
    # determine the annealing schedule
    BATCH_SIZE = args.batch_size
    num_iter = args.num_iter
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
    # positions = np.random.rand(n, 2)
    # x = torch.tensor(positions, requires_grad=True)
    x = torch.rand(n, requires_grad=True)
    y = torch.rand(n, requires_grad=True)
    # print(f"x.shape: {x.shape}") # (882, 2)
    
    # stress = compute_stress(positions, d)
    # print(f"initial stress: {stress}")

    my_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # ========== Training ============
    start = datetime.datetime.now()

    rnd_node_analysis_2 = [0]*dataset.get_node_count()
    for idx_c, c in enumerate(schedule):
        lr = 0.4
        loss_fn = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD([x,y], lr=lr)
        for batch_idx, (i, j, w, dis) in enumerate(my_dataloader): # batch size = 2
            print(batch_idx, " / ", dataset.__len__())
            rnd_node_analysis_2[i-1] = rnd_node_analysis_2[i-1] + 1
            rnd_node_analysis_2[j-1] = rnd_node_analysis_2[j-1] + 1

            dx = x[i-1] - x[j-1]
            dy = y[i-1] - y[j-1]
            d = torch.sqrt(torch.pow(dx,2) + torch.pow(dy,2))
            dis_ten = torch.tensor([float(dis)])
            loss = loss_fn(d, dis_ten)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # w_choose = torch.min(w) # choose the minimum w in the batch. This is different from the original paper. 
            # wc = w * c
            # wc = torch.min(wc, torch.ones_like(wc))
            # if (wc > 1):
            #     wc = 1
            # lr = torch.min(wc / (4 * w)) # really need this "/4" -> check the graph drawing paper 
            
            # stress = q(x[i-1], x[j-1], dis)
            # stress.backward()

            # if (batch_idx % 100 == 0):
            #     print(f"i: {i}, j: {j}, w: {w}, dis: {dis}, lr: {lr}, wc: {wc}, stress: {stress}")
            #     # print(f"stress: {stress}")
            #     print(f"x.data: {x.data}")
            #     print(f"torch.max(x.grad.data): {torch.max(x.grad.data)}")
            # elif (batch_idx == 2001):
            #     sys.exit()

            # x.data.sub_(lr * x.grad.data) # lr set to be a vector?
            # x.grad.data.zero_()
            
    end = datetime.datetime.now()
    print(f"Time: {end - start}")

    # Draw the Visualization Graph
    x_np = x.detach().numpy()
    y_np = y.detach().numpy()
    # print(f"x_np.shape: {x_np.shape}")
    # print(f"x_np: {x_np}")

    # NOTE: why is the result so wrong?
    draw_svg(x_np, y_np, dataset, f"out")

    print(dataset.rnd_path_analysis)
    print(dataset.rnd_node_analysis)
    print(rnd_node_analysis_2)
    
    # Compute the Total Stress
    # stress_total = compute_stress(x_np, d)
    # print(f"stress_total: {stress_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SGD Implementation for Graph Drawing")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--file', type=str, default='DRB1-3123.og', help='odgi variation graph')
    args = parser.parse_args()
    print(args)
    main(args)
