# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# Run with command 'env LD_PRELOAD=libjemalloc.so.2 PYTHONPATH=<lib dir of odgi-build> python3 batch_sgd.py --batch_size 1 --num_iter 15'

import torch
from odgi_ffi import *

class OdgiDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        self.g = odgi_load_graph(file_name)

        assert odgi_min_node_id(self.g) == 1
        assert odgi_max_node_id(self.g) == self.get_node_count()

        self.rnd_node_gen = odgi_create_rnd_node_generator(self.g)

        self.path_names = []
        odgi_for_each_path_handle(self.g, lambda p: self.path_names.append(odgi_get_path_name(self.g, p)))

        self.sizes = []
        odgi_for_each_path_handle(self.g, lambda p: self.sizes.append(odgi_get_step_in_path_count(self.g, p)))

        self.w_max = 1                                  # w_max when nodes next to each other
        self.w_min = 1 / ((max(self.sizes)-1)**2)       # w_min when nodes on ends of longest path
        return

    def __getitem__(self, index): # return i, j, w, dis[i, j]
        node_pack = odgi_get_random_node_pack(self.rnd_node_gen)
        id_node_a = odgi_RNP_get_id_n0(node_pack)
        id_node_b = odgi_RNP_get_id_n1(node_pack)
        d = float(odgi_RNP_get_distance(node_pack))
        w = 1.0/float(d**2)
        vis_p_a = odgi_RNP_get_vis_p_n0(node_pack)
        vis_p_b = odgi_RNP_get_vis_p_n1(node_pack)
        return id_node_a, id_node_b, vis_p_a, vis_p_b, w, d

    def __len__(self):
        return sum(self.sizes) * 10         # similar to odgi default

    def get_path(self, idx):
        assert 0 <= idx < self.get_path_count()
        path_name = self.path_names[idx]
        return odgi_get_path_handle(self.g, path_name)

    def get_node_count(self):
        return odgi_get_node_count(self.g)

    def get_path_count(self):
        return odgi_get_path_count(self.g)

    # def get_node_ids_in_path(self, path, sidx_list):
    #     assert isinstance(sidx_list, list)
    #     prev_idx = None
    #     ids = []
    #     step = self.g.path_begin(path)
    #     for sidx in sidx_list:
    #         delta = sidx
    #         if prev_idx != None:
    #             delta = delta - prev_idx
    #         prev_idx = sidx

    #         for i in range(delta):
    #             assert self.g.has_next_step(step) == True
    #             step = self.g.get_next_step(step)
    #         ids.append(self.g.get_id(self.g.get_handle_of_step(step)))
    #     return ids

    # def get_step_count_in_path(self, path):
    #     return self.g.get_step_count(path)

class OdgiInterface:
    def generate_layout_file(graph, coords, file_name):
        odgi_generate_layout_file(graph, coords, file_name)
