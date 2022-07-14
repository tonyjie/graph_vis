# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# Run with command 'env LD_PRELOAD=libjemalloc.so.2 PYTHONPATH=<lib dir of odgi-build> python3 batch_sgd.py --batch_size 1 --num_iter 15'

import random
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

from odgi_dataset import OdgiDataset, OdgiInterface


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
    # # ========== Load Graph, Get Constraints ============
    dataset = OdgiDataset(args.file)

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
    x = torch.rand([n,2,2], requires_grad=True)
    # print(f"x.shape: {x.shape}") # (882, 2)
    
    # stress = compute_stress(positions, d)
    # print(f"initial stress: {stress}")

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
            stress.backward()

            # if (batch_idx % 100 == 0):
            #     print(f"i: {i}, j: {j}, w: {w}, dis: {dis}, lr: {lr}, wc: {wc}, stress: {stress}")
            #     # print(f"stress: {stress}")
            #     print(f"x.data: {x.data}")
            #     print(f"torch.max(x.grad.data): {torch.max(x.grad.data)}")
            # elif (batch_idx == 2001):
            #     sys.exit()

            x.data.sub_(lr * x.grad.data) # lr set to be a vector?
            x.grad.data.zero_()
            
    end = datetime.datetime.now()
    print(f"Time: {end - start}")

    # Draw the Visualization Graph
    x_np = x.detach().numpy()
    # print(f"x_np.shape: {x_np.shape}")
    # print(f"x_np: {x_np}")

    OdgiInterface.generate_layout_file(dataset.g, x_np, "output/" + args.file + ".lay")
    draw_svg(x_np, dataset, f"out")

    # Compute the Total Stress
    # stress_total = compute_stress(x_np, d)
    # print(f"stress_total: {stress_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SGD Implementation for Graph Drawing")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--file', type=str, default='tiny_pangenome.og', help='odgi variation graph')
    args = parser.parse_args()
    print(args)
    main(args)
