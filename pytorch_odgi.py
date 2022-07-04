# env LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=~/odgi/lib python pytorch_odgi.py
import odgi
import torch
import numpy as np
from torch import nn
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import datetime
import argparse
import imageio
import scipy.sparse.csgraph as csgraph
import scipy.io as io

def draw(pos_changes, nonzero_adj, output_dir, DRAW_INTERVAL):
    # remove all the .png file in output_dir
    # os.system(f"rm {output_dir}/*.png")

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

        for i, j in zip(*nonzero_adj):
            lines.append([positions[i], positions[j]])

        # for i in range(nonzero_adj.shape[0]):
        #     start, end = nonzero_adj[i][0], nonzero_adj[i][1]
            
        #     lines.append([positions[start], positions[end]])
        

        lc = mc.LineCollection(lines, linewidths=1, colors='k', alpha=.5)
        ax.add_collection(lc)

        # plot the node and text
        # plt.plot(positions[:,0], positions[:,1], marker='o')
        # for i in range(positions.shape[0] // 2):
        #     plt.text((positions[2*i,0]+positions[2*i+1,0])/2, (positions[2*i,1]+positions[2*i+1,1])/2, str(i))

        fig.savefig(f"{output_dir}/{idx * DRAW_INTERVAL}.png")
        frames.append(imageio.imread(f"{output_dir}/{idx * DRAW_INTERVAL}.png"))
    
    imageio.mimsave(f"{output_dir}/animation.gif", frames, duration=0.1)

class PlaceEngine(nn.Module):
    '''
    @brief Graph 2D Layout Engine. It contains the parameters ([X, Y] coordinates of all nodes) to be updated. 
    '''

    def __init__(self, num_nodes):
        '''
        @brief initialization
        @param num_nodes: number of nodes in the graph
        '''
        super().__init__()
        self.num_nodes = num_nodes
        pos = torch.empty(num_nodes, 2)
        self.pos = nn.Parameter(data=pos, requires_grad=True)
        nn.init.uniform_(self.pos) # can try other initialization methods: kaiming_uniform; xavier_uniform

        # self.dist = dist # [num_nodes, num_nodes]
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
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
        stress = torch.sum(stress_matrix)
        
        return stress



def main(args):
    # Problems: dist(viz point A, viz point B) can vary depending on the diffrent paths. 
    # e.g. dis(A, B) = 2 on Path X; but dis(A, B) = 5 on Path Y. 
    STEPS = args.steps
    DRAW_INTERVAL = args.draw_interval
    LOG_INTERVAL = args.log_interval

    g = odgi.graph()
    g.load(args.input_file)
    # g.load('data_pangenome/DRB1-3123/DRB1-3123.og')

    num_nodes = g.get_node_count() * 2 # number of visualization points: 2 * graph nodes. One node has 2 ends. 
    num_paths = g.get_path_count()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"==== input_file: {args.input_file}; num_nodes: {num_nodes}; num_paths: {num_paths}; Device: {device} ====")

    # Get an adjacency matrix [num_nodes, num_nodes]. 
    # If there's a connection for two points in any path, it is True. 
    # Used for drawing the graph. Determine if we draw a line between two points.
    adj_matrix = np.zeros((num_nodes, num_nodes)) # all False initially

    def fill_adj_matrix(path_handle):
        '''
        @brief fill the adj_matrix. If there's a connection for two points in the path, set it as True. 
        @param path_handle: handle of the path
        '''
        step_handles = []
        g.for_each_step_in_path(path_handle, lambda s: step_handles.append(g.get_handle_of_step(s)))
        
        step_id = np.zeros(len(step_handles), dtype=int)
        point_id = np.zeros(2 * len(step_handles), dtype=int)
        for idx, step_handle in enumerate(step_handles):
            step_handle_id = g.get_id(step_handle) - 1
            step_id[idx] = step_handle_id
            point_id[idx*2] = step_handle_id * 2
            point_id[idx*2+1] = step_handle_id * 2 + 1
        
        print(f"point_id: {point_id}")
        for i in range(len(point_id) - 1):
            adj_matrix[point_id[i], point_id[i+1]] = True
            # print(f"{point_id[i]}, {point_id[i+1]} are connected")

    g.for_each_path_handle(lambda p: fill_adj_matrix(p)) # can be combined into `get_dist`
    
    nonzero_adj = np.nonzero(adj_matrix) # (start, end). 
    # nonzero_adj[0] is a numpy array (length = num_nodes), record the starting point. 
    # nonzero_adj[1] is a numpy array (length = num_nodes), record the ending point.
    
    num_lines = len(nonzero_adj[0])
    print(f"num_lines: {num_lines}")

    Dist_paths = list() # Distance matrix for different paths

    def get_dist(path_handle):
        '''
        @brief Get the distance matrix for a path.
            The path has N steps, each step has 2 ends -> N*2 viz points are defined. 
        @param path_handle: Path handle
        @return: Distance matrix [N*2, N*2]
        This is lambda function applied on each path_handle. 
        '''
        
        step_handles = []
        g.for_each_step_in_path(path_handle, lambda s: step_handles.append(g.get_handle_of_step(s)))
        
        step_length = np.zeros(len(step_handles), dtype=int)
        step_id = np.zeros(len(step_handles), dtype=int)
        point_id = np.zeros(2 * len(step_handles), dtype=int)    

        for idx, step_handle in enumerate(step_handles):
            step_handle_id = g.get_id(step_handle) - 1
            step_id[idx] = step_handle_id
            if g.get_is_reverse(step_handle) == False: # '+' strand
                point_id[idx*2] = step_handle_id * 2
                point_id[idx*2+1] = step_handle_id * 2 + 1
            else:                                      # '-' strand             
                point_id[idx*2] = step_handle_id * 2 + 1
                point_id[idx*2+1] = step_handle_id * 2
            step_length[idx] = g.get_length(step_handle)

        print(f"step_id: {step_id}")
        print(f"step_length: {step_length}")

        Dist = np.zeros((num_nodes, num_nodes)) # Distance Matrix

        for i, pi in enumerate(point_id):
            for j, pj in enumerate(point_id):
                if i < j:
                    pair_dist = np.sum(step_length[(i+1)//2 : (j+1)//2])
                    if pair_dist == 0: # end of one node is connected to the start of the next node. If set to zero, it will be masked. 
                        pair_dist = 1e-1 # set a very small value
                    Dist[pi, pj] = pair_dist
                    # print(f"{i}, {j}: {step_length[(i+1)//2 : (j+1)//2]} -> {pair_dist}")
        return Dist



    
    
    if args.save:
        Dist_paths_arr = np.load(os.path.dirname(args.input_file) + '/Dist_paths.npy')
    else:
        g.for_each_path_handle(lambda p: Dist_paths.append(get_dist(p)))
        Dist_paths_arr = np.array(Dist_paths)
        np.save(os.path.dirname(args.input_file) + '/Dist_paths.npy', Dist_paths_arr)

    print(Dist_paths_arr.shape) # [num_paths, num_nodes, num_nodes]

    print("====== Finish Computing Distance Matrix for different paths ======")
    # [Problem] connected code length is zero, it doesn't take into consideration when training! e.g. Node1 and Node2 are connected. 
    
    # [Problem] some .gfa file has abnormal pattern. DRB1-3123: point_id: [9906 9907 9904 ...   23   10   11]

    










    # mat_data = io.loadmat(args.input_file)
    # graph = mat_data['Problem']['A'][0][0]
    # dist = csgraph.shortest_path(graph, directed=False, unweighted=True)
    # print(f"dist.shape: {dist.shape}") # (5, 5)
    # print(f"dist: {dist}")
    

    # num_nodes = dist.shape[0]

    # dis = np.random.randn(num_nodes, num_nodes)
    # dist = torch.tensor(dist, dtype=torch.float32)
    # dist = dist.to(device)

    
    dist_paths = torch.tensor(Dist_paths_arr, dtype=torch.float32)
    dist_paths = dist_paths.to(device)

    mod = PlaceEngine(num_nodes)
    mod = mod.to(device)
    # print(mod.pos)

    pos_changes = np.zeros((STEPS//DRAW_INTERVAL, num_nodes, 2), dtype=np.float32)

    stress_rec = np.zeros((STEPS//LOG_INTERVAL,), dtype=np.float32)
    # ====== Training ======
    start = datetime.datetime.now()
    for i in range(STEPS):
        # switch distance matrix between paths different paths in each step
        stress = mod.gradient_step(dist_paths[i % num_paths])
        mod.scheduler.step() # learning rate scheduler

        if i % LOG_INTERVAL == 0:
            print(f"step {i}/{STEPS}: {stress}")
            stress_rec[i//LOG_INTERVAL] = stress.item()

        if args.draw and i % DRAW_INTERVAL == 0:
            pos = mod.pos.cpu().detach().numpy()
            pos_changes[i // DRAW_INTERVAL] = pos

    end = datetime.datetime.now()

    print(f"==== Training time: {end - start}; Step: {STEPS}; Device: {device} ====")

    # === draw learning curve ===
    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(0, STEPS, STEPS//LOG_INTERVAL), stress_rec)
    # plt.savefig(f"{os.path.dirname(args.input_file)}/learning_curve.png")


    if args.draw:
        draw(pos_changes, nonzero_adj, os.path.dirname(args.input_file), DRAW_INTERVAL)

    








# Problems: dist(viz point A, viz point B) can vary depending on the diffrent paths. 
# e.g. dis(A, B) = 2 on Path X; but dis(A, B) = 5 on Path Y. 

# g = odgi.graph()
# g.load("tiny_pangenome.og")
# # g.load('data_pangenome/DRB1-3123/DRB1-3123.og')

# num_nodes = g.get_node_count() * 2 # number of visualization points: 2 * graph nodes. One node has 2 ends. 
# print(f"num_nodes: {num_nodes}")

# num_paths = g.get_path_count()
# print(f"num_paths: {num_paths}")

# Dist_paths = list() # Distance matrix for different paths

# def get_dist(path_handle):
#     '''
#     @brief Get the distance matrix for a path.
#         The path has N steps, each step has 2 ends -> N*2 viz points are defined. 
#     @param path_handle: Path handle
#     @return: Distance matrix
#     '''
    
#     step_handles = []
#     g.for_each_step_in_path(path_handle, lambda s: step_handles.append(g.get_handle_of_step(s)))
    
#     step_length = np.zeros(len(step_handles), dtype=int)
#     step_id = np.zeros(len(step_handles), dtype=int)
#     point_id = np.zeros(2 * len(step_handles), dtype=int)    

#     for idx, step_handle in enumerate(step_handles):
#         step_handle_id = g.get_id(step_handle) - 1
#         step_id[idx] = step_handle_id
#         point_id[idx*2] = step_handle_id * 2
#         point_id[idx*2+1] = step_handle_id * 2 + 1
#         step_length[idx] = g.get_length(step_handle)

#     print(f"step_id: {step_id}")
#     print(f"step_length: {step_length}")

#     Dist = np.zeros((num_nodes, num_nodes)) # Distance Matrix

#     for i, pi in enumerate(point_id):
#         for j, pj in enumerate(point_id):
#             if i < j:
#                 # print(f"{i}, {j}: {step_length[i//2 : j//2]}")
#                 pair_dist = np.sum(step_length[i//2 : j//2])

#                 Dist[pi, pj] = pair_dist
#     return Dist


# g.for_each_path_handle(lambda p: Dist_paths.append(get_dist(p)))

# print(Dist_paths)

# Dist_paths_arr = np.array(Dist_paths)
# print(Dist_paths_arr.shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGD Implementation for Graph Drawing using Pytorch")
    parser.add_argument("input_file", type=str, help="input graph name (.og)")    
    # parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--steps', type=int, default=100, help='number of steps')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--draw', action='store_true', default=False, help='Draw Animation')
    parser.add_argument('--draw_interval', type=int, default=1, help='Draw Animation Interval (Steps)')
    parser.add_argument('--save', action='store_true', default=False, help='Use already saved .npy file.')

    args = parser.parse_args()
    print(args)
    main(args)






# ========= Try on ODGI python interface ==========
# g = odgi.graph()
# g.load("tiny_pangenome.og")

# path_names = []
# g.for_each_path_handle(lambda p: path_names.append(g.get_path_name(p)))

# sizes = []
# g.for_each_path_handle(lambda p: sizes.append(g.get_step_count(p)))

# path_handle = []
# g.for_each_path_handle(lambda p: path_handle.append(p))

# step_handles = []
# g.for_each_step_in_path(path_handle[0], lambda s: step_handles.append(g.get_handle_of_step(s)))

# step_length = []


# sequence = ""
# for step_handle in step_handles:
#     sequence += g.get_sequence(step_handle)
#     step_length.append(g.get_length(step_handle))

# g.for_each_handle(lambda h: print(f"{g.get_length(h)}: {g.get_sequence(h)}"))

# print(f"path_names: {path_names}")
# print(f"sizes: {sizes}")
# print(f"path_handle: {path_handle}")
# print(f"step_handle: {step_handles}")

# print(f"sequence: {sequence}")
# print(f"step_length: {step_length}")

# for step_handle in step_handles:
#     print(g.get_id(step_handle)) # 1 3 5 6 8 9 11 12 14 15