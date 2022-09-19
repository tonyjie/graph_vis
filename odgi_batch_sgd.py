# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# env LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=~/odgi_niklas/lib python odgi_batch_sgd.py --file <input_file> --batch_size 1 --num_iter 30
import datetime
import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from odgi_dataset import OdgiTorchDataset, OdgiInterface
import sys

def draw_svg(x, output_name):
    # print("drawing visualization")
    # Draw SVG Graph with edge
    # ax = plt.axes()
    fig, ax = plt.subplots()

    xmin = x.min()
    xmax = x.max()
    edge = 0.1 * (xmax - xmin)
    ax.set_xlim(xmin-edge, xmax+edge)
    ax.set_ylim(xmin-edge, xmax+edge)

    for p in x:
        plt.plot(p[:,0], p[:,1], '-', linewidth=1)

    # for i in range(gdata.get_node_count()):
    #     plt.text(np.mean(x[i,:,0]), np.mean(x[i,:,1]), i+1)

    plt.savefig(output_name, format='png', dpi=1000)

def stress_fn(pos, dist):
    '''
    @brief: compute the overall stress given the positions and Distance Matrix.
    @param: positions. [num_nodes, 2]
    @param: dist: [num_paths, num_nodes, num_nodes]
    @return: stress. 
    '''

    num_nodes = pos.shape[0]
    copy1 = torch.reshape(pos, (1, num_nodes, 2))
    copy2 = torch.reshape(pos, (num_nodes, 1, 2))
    broadcasted1 = torch.broadcast_to(copy1, (num_nodes, num_nodes, 2))
    broadcasted2 = torch.broadcast_to(copy2, (num_nodes, num_nodes, 2))
    diff = broadcasted1 - broadcasted2
    pred_dist = torch.norm(diff, dim=2).reshape((num_nodes, num_nodes))

    mask = dist.ne(0)

    # pred_dist = torch.where(mask, pred_dist, dist)
    stress_matrix = torch.where(mask, torch.square((pred_dist - dist) / dist), dist) # [num_nodes, num_nodes]. Actually, this involves with redundant computation. (The matrix is symmetric)
    stress = torch.sum(stress_matrix)

    return stress


def compute_stress(pos, Dist_paths):# Wrong!!!
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
        pred_dist = np.where(mask, pred_dist, Dist) # This line will change pred_dist, which is NOT correct! 
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

    # breakpoint()
    # change 1e-9 to 0. We don't compute these terms for stress. 
    Dist_paths_arr[np.where(Dist_paths_arr == 1e-9)] = 0
    dist = torch.tensor(Dist_paths_arr, dtype=torch.float32)

    # # ========== Load Graph, Get Constraints ============
    dataset = OdgiTorchDataset(args.file)

    print(f"Steps_per_iter: {len(dataset)}")


    n = dataset.get_node_count()

    print(f"node_count: {n}")
    print(f"update per node: {len(dataset) / n}")

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

    stress_rec = np.zeros(num_iter + 1)
    # ========== Initialize the Positions ============

    x = torch.rand([n,2,2], requires_grad=True)


    initial_stress = stress_fn(x.reshape(n*2, 2), dist)

    # initial_stress = compute_stress(x.detach().numpy().reshape(n*2,2), Dist_paths_arr)
    print(f"initial stress: {initial_stress:.2e}")
    draw_svg(x.detach().numpy(), f"initial")
    stress_rec[0] = initial_stress

    my_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)       # is always shuffled anyways
    # ========== Training ============
    
    start = datetime.datetime.now()
    for idx_c, c in enumerate(schedule):
        # print("Computing iteration", idx_c+1, "of", num_iter)
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

        x_np = x.detach().numpy()
        draw_svg(x_np, f"{os.path.dirname(args.file)}/iter{idx_c}.png")
        OdgiInterface.generate_layout_file(dataset.get_graph(), x_np, f"{os.path.dirname(args.file)}/iter{idx_c}.lay")

        stress = stress_fn(x.reshape(n*2,2), dist)
        # stress = compute_stress(x_np.reshape(n*2,2), Dist_paths_arr)
        print(f"Iteration {idx_c+1}: Stress = {stress:.2e}")
        stress_rec[idx_c+1] = stress

    end = datetime.datetime.now()
    print(f"Time: {end - start}")

    # Draw the Visualization Graph
    # x_np = x.detach().numpy()

    # OdgiInterface.generate_layout_file(dataset.get_graph(), x_np, "output/lil.lay")
    # draw_svg(x_np, f"out")

    # === draw learning curve ===
    fig, ax = plt.subplots()
    ax.plot(stress_rec)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Stress')
    ax.set_yscale('log')
    ax.set_title('Stress Curve: Pairwise Update')
    plt.tight_layout()
    plt.savefig(f"{os.path.dirname(args.file)}/stress_curve_odgi_batch_sgd.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SGD Implementation for Graph Drawing")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--file', type=str, default='tiny_pangenome.og', help='odgi variation graph')
    args = parser.parse_args()
    print(args)
    main(args)