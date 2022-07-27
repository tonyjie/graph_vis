# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# Run with command 'env LD_PRELOAD=libjemalloc.so.2 PYTHONPATH=<lib dir of odgi-build> python3 batch_sgd.py --batch_size 1 --num_iter 15'
# env LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=~/odgi_niklas/lib python odgi_model.py --file DRB1-3123.og --batch_size=100 --num_iter=30 --create_iteration_figs
import argparse
import sys
import os
import math
from datetime import datetime
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from odgi_dataset import OdgiDataloader, OdgiInterface

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


def draw_svg(x, gdata, output_name):
    print('Drawing visualization "{}"'.format(output_name))
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

    plt.axis("off")
    plt.savefig(output_name + '.png', format='png', dpi=1000, bbox_inches="tight")
    plt.close()


def main(args):
    if (args.stress):
        Dist_paths_arr = np.load(os.path.dirname(args.file) + '/Dist_paths.npy')
        print(f"==== Finish Loading Dist_paths_arr: {Dist_paths_arr.shape} ====") # [num_paths, num_nodes, num_nodes]


    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"==== Device: {device}; Dataset: {args.file} ====")

    data = OdgiDataloader(args.file, batch_size=args.batch_size)
    # data.set_batch_size(args.batch_size)

    n = data.get_node_count()
    num_iter = args.num_iter

    w_min = data.w_min
    w_max = data.w_max
    epsilon = 0.01              # default value of odgi

    eta_max = 1/w_min
    eta_min = epsilon/w_max
    lambd = math.log(eta_min / eta_max) / (num_iter - 1)
    schedule = []
    for t in range(num_iter+1):
        eta = eta_max * math.exp(lambd * t)
        schedule.append(eta)




    print(f"len(data): {data.steps_in_iteration()}") # 350590 for DRB1-3123.og
    # torch.set_num_interop_threads(1)
    # print(f"==== Config: num_threads: {torch.get_num_threads()}; num_interop_threads: {torch.get_num_interop_threads()} ====")

    x = torch.rand([n,2,2], dtype=torch.float64, device=device)

    if (args.stress):
        initial_stress = compute_stress(x.cpu().detach().numpy().reshape(n*2,2), Dist_paths_arr)
        print(f"initial stress: {initial_stress:.2e}")


    start = time.time()
    
    compute_time = 0
    transfer_time = 0

    # ***** Interesting NOTE: I think odgi runs one iteration more than selected with argument
    for iteration, eta in enumerate(schedule):
        print("Computing iteration", iteration + 1, "of", num_iter + 1, eta)
        
        for batch_idx, (i, j, vis_p_i, vis_p_j, _w, dis) in enumerate(data):
            # breakpoint()

            # pytorch model as close as possible to odgi implementation

            # compute weight w in PyTorch model (here); don't use computed weight of dataloader
            # dataloader computes it as w = 1 / dis^2
            # ***** Interesting NOTE (found by Jiajie): ODGI uses as weight 1/dis; while original graph drawing paper uses 1/dis^2
            
            transfer_start = time.time()
            # to cuda
            dis = dis.to(device)
            if (device == torch.device("cuda")):
                torch.cuda.synchronize()
            transfer_end = time.time()

            compute_start = time.time()
            w = 1 / dis

            mu = eta * w
            mu_m = torch.min(mu, torch.ones_like(mu))

            x_i = x[i-1,vis_p_i,0]
            x_j = x[j-1,vis_p_j,0]
            y_i = x[i-1,vis_p_i,1]
            y_j = x[j-1,vis_p_j,1]

            dx = x_i - x_j
            dy = y_i - y_j

            # The following two lines will cause LPA & MHC to have different results with ODGI. However, the following lines follows the OGDI's strategies on avoiding NaN. 
            # not_zero = torch.ones_like(dx) * 1e-9
            # dx = torch.max(dx, not_zero)

            mag = torch.pow(torch.pow(dx,2) + torch.pow(dy,2), 0.5)

            not_zero = torch.ones_like(mag) * 1e-9
            mag_not_zero = torch.max(mag, not_zero)

            delta = mu_m * (mag - dis) / 2.0

            r = delta / mag_not_zero
            r_x = r * dx
            r_y = r * dy

            x[i-1, vis_p_i, 0] = x[i-1, vis_p_i, 0] - r_x
            x[j-1, vis_p_j, 0] = x[j-1, vis_p_j, 0] + r_x
            x[i-1, vis_p_i, 1] = x[i-1, vis_p_i, 1] - r_y
            x[j-1, vis_p_j, 1] = x[j-1, vis_p_j, 1] + r_y

            # if (torch.isnan(x).any()):
            #     # breakpoint()
            #     raise ValueError(f"Iter[{iteration}] Step[{batch_idx}]. x: {x} is NaN. mag: {mag}")

            if (device == torch.device("cuda")):
                torch.cuda.synchronize()
            compute_end = time.time()

            transfer_time += transfer_end - transfer_start
            compute_time += compute_end - compute_start

        if (args.stress):
            stress = compute_stress(x.cpu().detach().numpy().reshape(n*2,2), Dist_paths_arr)
            print(f"Iteration {iteration+1} Done.  Stress: {stress:.2e}")

        if ((iteration+1) % 5) == 0 and args.create_iteration_figs == True:
            x_np = x.cpu().clone().detach().numpy()
            draw_svg(x_np, data, f"{os.path.dirname(args.file)}/out_iter{iteration+1}")

    end = time.time()
    overall_time = end-start
    dataload_time = overall_time - compute_time - transfer_time
    print(f"Overall time {overall_time:.2f} sec; Dataloading time: {dataload_time:.2f} sec; Computation time: {compute_time:.2f} sec; Data Transfer time: {transfer_time:.2f} sec")

    # x_np = x.cpu().detach().numpy()
    # OdgiInterface.generate_layout_file(data.get_graph(), x_np, os.path.dirname(args.file) + "/layout.lay")
    # draw_svg(x_np, data, f"{os.path.dirname(args.file)}/out_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SGD Implementation for Graph Drawing")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--file', type=str, default='tiny_pangenome.og', help='odgi variation graph')
    parser.add_argument('--create_iteration_figs', action='store_true', help='create at each 5 iteration a figure of the current state')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--stress', action='store_true', default=False, help='compute stress')
    args = parser.parse_args()
    print(args)
    main(args)