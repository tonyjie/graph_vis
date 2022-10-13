# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# env LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=~/odgi_niklas/lib python odgi_sgd_vector_dataload.py data_pangenome/lil/lil.og --num_iter 30 --batch_size 100 --cuda --log_interval 20 --draw --lay
import time
import argparse
import torch
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from odgi_dataset import OdgiTorchDataset, OdgiInterface, OdgiDataloader

def draw(pos_changes, output_dir):
    def draw_one_graph(pos, idx, output_dir):
        fig, ax = plt.subplots()

        xmin = pos.min()
        xmax = pos.max()
        edge = 0.1 * (xmax - xmin)
        ax.set_xlim(xmin-edge, xmax+edge)
        ax.set_ylim(xmin-edge, xmax+edge)
        ax.set_title(f"Iter {idx}")

        pos = pos.reshape((pos.shape[0]//2, 2, 2))

        for p in pos:
            plt.plot(p[:,0], p[:,1], '-', linewidth=1)
        plt.savefig(f"{output_dir}/iter{idx}.png")
        plt.close(fig)
        frames.append(imageio.v2.imread(f"{output_dir}/iter{idx}.png"))

    frames = list()
    for idx, pos in enumerate(pos_changes):
        draw_one_graph(pos, idx, output_dir)

    imageio.mimsave(f"{output_dir}/iter_animate.gif", frames, duration=0.1)


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"==== Device: {device}; Dataset: {args.input_file} ====")

    # print(f"torch.get_num_threads(): {torch.get_num_threads()}") # 20
    # torch.set_num_threads(32)

    # Dist_paths_arr = np.load(os.path.dirname(args.input_file) + '/Dist_paths.npy')
    # print(f"==== Finish Loading Dist_paths_arr: {Dist_paths_arr.shape} ====") # [num_paths, num_nodes, num_nodes]

    # dist = torch.tensor(Dist_paths_arr, dtype=torch.float32)
    # make it symmetric
    # dist = dist.transpose(1,2) + dist



    # change 1e-9 to 0. We don't compute these terms for stress. 
    # Dist_paths_arr[np.where(Dist_paths_arr == 1e-9)] = 0


    # # ========== Load Graph, Get Constraints ============
    data = OdgiDataloader(args.input_file, batch_size=args.batch_size, nthreads=args.nthreads) # set batch size


    num_pangenome_nodes = data.get_node_count()
    num_nodes = num_pangenome_nodes * 2

    print(f"==== num_pangenome_nodes: {num_pangenome_nodes}; steps_per_iteration: {data.steps_in_iteration()} ====")

    # ========== Determine the annealing schedule (Step Size). Parameter Setting. ===========
    w_min = data.w_min
    w_max = data.w_max

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
    pos = torch.empty((num_nodes, 2), dtype=torch.float32)
    nn.init.uniform_(pos)

    
    def one_step(pos, dist, c):
        '''
        @brief: One iteration step: update all the nodes in the graph. 
        @param: pos: [num_nodes, 2]. The current position of the nodes.
        @param: dist: [num_paths, num_nodes, num_nodes]
        @param: c: constant. step size (annealing rate)
        '''
        
        # need to add the cooling phase: chooose the close one

        # maybe we have more efficient way to do this. Currently, there will be some 0 for dis_vec
        num_nodes = pos.shape[0]
        index = torch.randperm(num_nodes)
        index_0 = index[0 : num_nodes//2]
        index_1 = index[num_nodes//2 : (num_nodes//2) * 2]       

        pos_0 = pos[index_0]
        pos_1 = pos[index_1]

        path = torch.randint(0, dist.shape[0], size=(num_nodes//2,)) # randomly choose one path

        dis_vec = dist[path, index_0, index_1]
        w_vec = torch.where(dis_vec != 0, 1 / dis_vec, torch.zeros_like(dis_vec))
        wc_vec = torch.min(w_vec * c, torch.ones_like(w_vec))

        diff = pos_0 - pos_1
        mag = torch.norm(diff, dim=1)
        mag = torch.max(mag, torch.ones_like(mag)*1e-9) # avoid mag = 0, will cause NaN
        r = (dis_vec - mag) / 2
        update = torch.unsqueeze(wc_vec * r / mag, dim=1) * diff

        pos[index_0] += update
        pos[index_1] -= update
    
        if (torch.isnan(pos).any()):
            raise ValueError(f"nan found in pos: {pos}\n update: {update}\n dis_vec: {dis_vec} \n w_vec: {w_vec}\n wc_vec: {wc_vec} \n mag: {mag} \n index_0: {index_0} \n index_1: {index_1} \n path: {path}")

    
    # ========== Training ============
    # dist = dist.to(device)
    pos = pos.to(device)

    pos_changes = np.zeros((args.num_iter, num_nodes, 2), dtype=np.float32)

    start = time.time()

    dataload_total = 0
    transfer_total = 0
    compute_total = 0


    for iter, eta in enumerate(schedule):
        transfer_iter = 0
        compute_iter = 0
        dataload_iter = 0
        dataload_start = time.time()

        for batch_idx, (vis_i, vis_j, dis) in enumerate(data): # (i,j) start from 1; vis_p_i, vis_p_j is in {0,1}
            dataload_iter += time.time() - dataload_start

            if (device == torch.device("cuda")):
                torch.cuda.synchronize()
            transfer_start = time.time()
            dis = dis.to(device)
            if (device == torch.device("cuda")):
                torch.cuda.synchronize()
            transfer_iter += time.time() - transfer_start

            compute_start = time.time()
            
            with torch.no_grad():
                w = 1 / dis # torch.clamp in-place. 
                mu = torch.min(w * eta, torch.ones_like(w)) # torch.ones_like() move out. Shape is consistent. 
                diff = pos[vis_i] - pos[vis_j] # maybe cpu -> gpu for those index is time-consuming?
                mag = torch.norm(diff, dim=1)
                mag = torch.max(mag, torch.ones_like(mag)*1e-9) # avoid mag = 0, will cause NaN
                r = (dis - mag) / 2
                update = torch.unsqueeze(mu * r / mag, dim=1) * diff
                pos[vis_i] += update
                pos[vis_j] -= update # memory contention?

            if (device == torch.device("cuda")):
                torch.cuda.synchronize()
            compute_iter += time.time() - compute_start

            if batch_idx % args.log_interval == 0:
                print(f"Iteration[{iter}]: {batch_idx}/{data.steps_in_iteration() // args.batch_size}")

            dataload_start = time.time()

        dataload_total += dataload_iter
        transfer_total += transfer_iter
        compute_total += compute_iter

        

        print(f"====== Time breakdown for Iter[{iter}] dataload: {dataload_iter:.2e}, transfer: {transfer_iter:.2e}, compute: {compute_iter:.2e} =====")

        pos_changes[iter] = pos.cpu().detach().numpy()
        

    elapsed = time.time() - start
    print(f"==== Elapsed time: {elapsed:.2f}s ====")
    print(f"====== Time breakdown:  dataload: {dataload_total:.2e}, transfer: {transfer_total:.2e}, compute: {compute_total:.2e} =====")

    result_dir = os.path.join(os.path.dirname(args.input_file), f"batch_size={args.batch_size}")
    if args.lay or args.draw:
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

    # generate ODGI .lay file
    if args.lay:
        for idx, pos in enumerate(pos_changes):
            pos_reshape = pos.reshape(num_nodes//2,2,2)
            OdgiInterface.generate_layout_file(data.get_graph(), pos_reshape, f"{result_dir}/iter{idx}.lay")

    if args.draw:
        draw(pos_changes, result_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ODGI Vector Processing for Pairwise Update")
    parser.add_argument('input_file', type=str, default='tiny_pangenome.og', help='odgi variation graph')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=30, help='number of iterations')
    # parser.add_argument("--steps_per_iter", type=int, default=5, help="steps per iteration")
    parser.add_argument('--draw', action='store_true', default=False, help='draw the graph')
    parser.add_argument('--lay', action='store_true', default=False, help='generate .lay file for ODGI to draw')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--nthreads', type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args)
