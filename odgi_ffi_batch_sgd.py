# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# env LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=~/odgi_niklas/lib python odgi_ffi_batch_sgd.py --file <input_file> --batch_size 1 --num_iter 30
import random
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
import os

import matplotlib.pyplot as plt

from odgi_dataset import OdgiTorchDataset, OdgiInterface


def draw_svg(x, gdata, output_name):
    print("drawing visualization")
    # Draw SVG Graph with edge
    ax = plt.axes()

    xmin = x.min()
    xmax = x.max()
    edge = 0.1 * (xmax - xmin)
    ax.set_xlim(xmin-edge, xmax+edge)
    ax.set_ylim(xmin-edge, xmax+edge)

    for p in x:
        plt.plot(p[:,0], p[:,1], '-', linewidth=1)

    # for i in range(gdata.get_node_count()):
    #     plt.text(np.mean(x[i,:,0]), np.mean(x[i,:,1]), i+1)

    plt.savefig('output/' + output_name + '.png', format='png', dpi=1000)


def compute_stress(pos, Dist_paths):
    '''
    @brief: compute the overall stress given the positions and Distance Matrix. 
    @param: positions. [num_nodes, 2]
    @param: Dist_paths: [num_paths, num_nodes, num_nodes]. Only nonzero element matters (are connected). 
    @return: stress. We compare the viz_dis with real_dis on All paths.
    '''
    num_nodes = pos.shape[0]
    copy1 = np.reshape(pos, (1, num_nodes, 2))
    copy2 = np.reshape(pos, (num_nodes, 1, 2))
    broadcasted1 = np.broadcast_to(copy1, (num_nodes, num_nodes, 2))
    broadcasted2 = np.broadcast_to(copy2, (num_nodes, num_nodes, 2))
    diff = broadcasted1 - broadcasted2
    pred_dist = np.linalg.norm(diff, axis=2).reshape((num_nodes, num_nodes))

    stress = 0
    num_paths = Dist_paths.shape[0]
    for i in range(num_paths):
        Dist = Dist_paths[i, :, :]
        mask = np.not_equal(Dist, 0)
        pred_dist = np.where(mask, pred_dist, Dist)
        stress_matrix = np.where(mask, np.square((pred_dist - Dist) / Dist), Dist)
        stress += np.sum(stress_matrix)
    
    return stress

def q(t1, t2, dis):
    # one term in the stress function
    w = 1 / (dis**2)
    return torch.mean(w * ((torch.norm((t1 - t2), dim=1) - dis) ** 2))


def main(args):
    Dist_paths_arr = np.load(os.path.dirname(args.file) + '/Dist_paths.npy')
    print(f"==== Finish Loading Dist_paths_arr: {Dist_paths_arr.shape} ====") # [num_paths, num_nodes, num_nodes]


    # # ========== Load Graph, Get Constraints ============
    dataset = OdgiTorchDataset(args.file)

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


    # ========== Initialize the Positions ============

    x = torch.rand([n,2,2], requires_grad=True)


    initial_stress = compute_stress(x.detach().numpy().reshape(n*2,2), Dist_paths_arr)
    print(f"initial stress: {initial_stress}")

    my_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)       # is always shuffled anyways
    # ========== Training ============
    start = datetime.datetime.now()
    for idx_c, c in enumerate(schedule):
        print("Computing iteration", idx_c+1, "of", num_iter)
        for batch_idx, (i, j, vis_p_i, vis_p_j, w, dis) in enumerate(my_dataloader): # batch size = 2
            # print(idx_c, ": ", batch_idx, " / ", dataset.__len__())
            # w_choose = torch.min(w) # choose the minimum w in the batch. This is different from the original paper. 
            wc = w * c
            wc = torch.min(wc, torch.ones_like(wc))
            # if (wc > 1):
            #     wc = 1
            lr = torch.min(wc / (4 * w)) # really need this "/4" -> check the graph drawing paper 
            
            stress = q(x[i-1,vis_p_i], x[j-1,vis_p_j], dis)
            # stress = q(x[(i-1)*2+vis_p_i], x[(j-1)*2+vis_p_j], dis)
            stress.backward()

            x.data.sub_(lr * x.grad.data) # lr set to be a vector?
            x.grad.data.zero_()

        stress = compute_stress(x.detach().numpy().reshape(n*2,2), Dist_paths_arr)
        print(f"Iteration {idx_c+1} Done.  Stress: {stress}")
            
    end = datetime.datetime.now()
    print(f"Time: {end - start}")

    # Draw the Visualization Graph
    x_np = x.detach().numpy()

    OdgiInterface.generate_layout_file(dataset.get_graph(), x_np, "output/lil.lay")
    draw_svg(x_np, dataset, f"out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SGD Implementation for Graph Drawing")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--file', type=str, default='tiny_pangenome.og', help='odgi variation graph')
    args = parser.parse_args()
    print(args)
    main(args)