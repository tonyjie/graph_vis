# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# env LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=${HOME}/odgi_niklas/lib python odgi_sgd_vector_dataload.py data_pangenome/lil/lil.og --num_iter 30 --batch_size 100 --cuda --log_interval 20 --draw --lay
'''
env CUDA_VISIBLE_DEVICES=2 LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=${HOME}/odgi_niklas/lib python odgi_real_batch_sgd.py ../data_pangenome/DRB1-3123/DRB1-3123.og --num_iter 30 --batch_size 1000 --cuda --log_interval 20 --draw
'''

# env CUDA_VISIBLE_DEVICES=2 LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=${HOME}/odgi_niklas/lib python odgi_real_batch_sgd.py ../data_pangenome/DRB1-3123/DRB1-3123.og --num_iter 30 --batch_size 5000 --cuda --log_interval 20 --draw

import time
import argparse
import torch
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from odgi_dataset import OdgiInterface, OdgiDataloader
import torch.profiler as profiler

PRINT_LOG = True
PROFILE = False
PYTORCH_PROFILE = False

class PlaceEngine(nn.Module):
    '''
    @brief: Graph 2D layout Engine. It contains the parameters ([X, Y] coordinates of all nodes) to be updated. 
    '''
    def __init__(self, num_nodes, lr_schedule):
        '''
        @brief initialization
        @param num_nodes: number of nodes in the graph
        '''
        super().__init__()
        self.lr_schedule = lr_schedule
        self.num_nodes = num_nodes
        pos = torch.empty(num_nodes, 2)
        self.pos = nn.Parameter(data=pos, requires_grad=True)
        nn.init.uniform_(self.pos) # can try other initialization methods: kaiming_uniform; xavier_uniform
    
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) # we'll set customized learning rate for each epoch
        
    def stress_fn(self, i, j, vis_p_i, vis_p_j, dis, iter):
        '''
        @brief Compute the stress function for batch_size pairs of nodes. This strictly follows the ODGI implementation. 
        '''
        diff = self.pos[(i-1)*2 + vis_p_i] - self.pos[(j-1)*2 + vis_p_j]
        mag = torch.norm(diff, dim=1)
        coeff = 1 / (4 * torch.max(dis, torch.tensor(self.lr_schedule[iter])))
        stress = coeff * (mag - dis) ** 2
        # sum up the stress for each node
        stress_sum = torch.sum(stress, dim=0)
        return stress_sum

    def gradient_step(self, i, j, vis_p_i, vis_p_j, dis, iter):
        self.optimizer.zero_grad()
        stress = self.stress_fn(i, j, vis_p_i, vis_p_j, dis, iter)
        stress.backward()
        # customized learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule[iter]
        self.optimizer.step()




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

# @profile
def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"==== Device: {device}; Dataset: {args.input_file} ====")

    # print(f"torch.get_num_threads(): {torch.get_num_threads()}") # 20
    torch.set_num_threads(1)

    # Dist_paths_arr = np.load(os.path.dirname(args.input_file) + '/Dist_paths.npy')
    # print(f"==== Finish Loading Dist_paths_arr: {Dist_paths_arr.shape} ====") # [num_paths, num_nodes, num_nodes]

    # dist = torch.tensor(Dist_paths_arr, dtype=torch.float32)
    # make it symmetric
    # dist = dist.transpose(1,2) + dist



    # change 1e-9 to 0. We don't compute these terms for stress. 
    # Dist_paths_arr[np.where(Dist_paths_arr == 1e-9)] = 0


    # # ========== Load Graph, Get Constraints ============
    data = OdgiDataloader(args.input_file, batch_size=args.batch_size) # set batch size


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

    print(f"==== eta_max: {eta_max}; eta_min: {eta_min} ====") # eta_max: 9610000.0; eta_min: 0.1
    print(f"==== w_max: {w_max}; w_min: {w_min} ====") # w_max: 1.0; w_min: 1.0405827263267429e-07

    lambd = np.log(eta_min / eta_max) / (num_iter - 1)
    eta = lambda t: eta_max*np.exp(lambd*t)

    # set up the schedule as an exponential decay
    schedule = []
    for i in range(num_iter):
        schedule.append(eta(i))

    # print(f"==== Schedule: {schedule} ====")
    # ==== Schedule: [9610000.0, 5098671.8762682425, 2705146.191659597, 1435239.626286049, 761479.2838970263, 404009.6783732206, 214350.96616667203, 113725.8317216375, 60338.26220648574, 32013.00734392382, 16984.788784519576, 9011.43235171553, 4781.09643044597, 2536.6536844579973, 1345.8444121517387, 714.049849539109, 378.8455656710329, 200.9999199933225, 106.6423141729574, 56.58003830320708, 30.019047872501766, 15.926875664919333, 8.450147037413661, 4.483301462017946, 2.378655887328116, 1.2620172607741673, 0.6695746009234478, 0.35524882276710884, 0.18848045595422055, 0.09999999999999998] ====


    # ========== Initialize the Positions ============
    # ODGI's initialization is not random actually. 
    # pos = torch.empty((num_nodes, 2), dtype=torch.float32, requires_grad=True, device=device)
    # nn.init.uniform_(pos)

    mod = PlaceEngine(num_nodes, schedule)
    mod = mod.to(device)

    
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
    # pos = pos.to(device)

    pos_changes = np.zeros((args.num_iter, num_nodes, 2), dtype=np.float32)

    start = time.time()

    if PROFILE: 
        dataload_total = 0
        transfer_total = 0
        compute_total = 0

    # with profiler.profile(
    #     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], 
    #                 with_stack=True, profile_memory=True, record_shapes=True) as prof:
    # enable Pytorch profiler
    for iter, eta in enumerate(schedule):
        if PROFILE:
            transfer_iter = 0
            compute_iter = 0
            dataload_iter = 0
            dataload_start = time.time()

        for batch_idx, (i, j, vis_p_i, vis_p_j, _w, dis) in enumerate(data): # (i,j) start from 1; vis_p_i, vis_p_j is in {0,1}
            if PROFILE:
                dataload_iter += time.time() - dataload_start
                if (device == torch.device("cuda")):
                    torch.cuda.synchronize()
                transfer_start = time.time()

            dis = dis.to(device)

            if PROFILE:
                if (device == torch.device("cuda")):
                    torch.cuda.synchronize()
                transfer_iter += time.time() - transfer_start

                compute_start = time.time()

            # ======== Wrap up within Pytorch nn.Module =======
            mod.gradient_step(i, j, vis_p_i, vis_p_j, dis, iter)



            # ========= Gradient Update Implementation ========
            # diff = pos[(i-1)*2 + vis_p_i] - pos[(j-1)*2 + vis_p_j]
            # mag = torch.norm(diff, dim=1)
            # coeff = 1 / (4 * torch.max(dis, torch.tensor(eta)))
            # stress = coeff * (mag - dis) ** 2
            # # sum up the stress for each node
            # stress_sum = torch.sum(stress, dim=0)
            # stress_sum.backward()
            # pos.data.sub_(eta * pos.grad.data)
            # pos.grad.data.zero_()



            # ======== Original Vector Implementation ==========
            # with torch.no_grad():
            #     w = 1 / dis # torch.clamp in-place. 
            #     mu = torch.min(w * eta, torch.ones_like(w)) # torch.ones_like() move out. Shape is consistent. 
            #     diff = pos[(i-1)*2 + vis_p_i] - pos[(j-1)*2 + vis_p_j] # maybe cpu -> gpu for those index is time-consuming?
            #     mag = torch.norm(diff, dim=1)
            #     mag = torch.max(mag, torch.ones_like(mag)*1e-9) # avoid mag = 0, will cause NaN
            #     r = (dis - mag) / 2
            #     update = torch.unsqueeze(mu * r / mag, dim=1) * diff
            #     pos[(i-1)*2 + vis_p_i] += update
            #     pos[(j-1)*2 + vis_p_j] -= update # memory contention?  

            if PROFILE:
                if (device == torch.device("cuda")):
                    torch.cuda.synchronize()
                compute_iter += time.time() - compute_start

            if PRINT_LOG:
                if batch_idx % args.log_interval == 0:
                    print(f"Iteration[{iter}]: {batch_idx}/{data.steps_in_iteration() // args.batch_size}")

            if PROFILE:
                dataload_start = time.time()

        if PROFILE:
            dataload_total += dataload_iter
            transfer_total += transfer_iter
            compute_total += compute_iter

            print(f"====== Time breakdown for Iter[{iter}] dataload: {dataload_iter:.2e}, transfer: {transfer_iter:.2e}, compute: {compute_iter:.2e} =====")

        pos_changes[iter] = mod.pos.cpu().detach().numpy()
        

    elapsed = time.time() - start
    print(f"==== Elapsed time: {elapsed:.2f}s ====")
    if PROFILE:
        print(f"====== Time breakdown:  dataload: {dataload_total:.2e}, transfer: {transfer_total:.2e}, compute: {compute_total:.2e} =====")

    # print("==== self_cpu_time_total ====")
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    # print("==== cpu_time_total ====")
    # print(prof.key_averages().table(sort_by="cpu_time_total"))
    # print("==== self_cuda_time_total ====")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    # print("==== cuda_time_total ====")
    # print(prof.key_averages().table(sort_by="cuda_time_total"))




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
    args = parser.parse_args()
    print(args)
    main(args)