# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# Run with command 'env LD_PRELOAD=libjemalloc.so.2 PYTHONPATH=<lib dir of odgi-build> python3 batch_sgd.py --batch_size 1 --num_iter 15'
# env LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=~/odgi_niklas/lib python odgi_cool_batch_sgd.py --file DRB1-3123.og --batch_size=100 --num_iter=31 --create_iteration_figs
import argparse
import sys
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from odgi_dataset import OdgiDataloader, OdgiInterface


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

def q(t1, t2, dis):
    # one term in the stress function
    w = 1 / dis
    return torch.mean(w * ((torch.norm((t1 - t2), dim=1) - dis) ** 2)) # change `mean` to `sum`

def main(args):
    data = OdgiDataloader(args.file)
    data.set_batch_size(args.batch_size)

    n = data.get_node_count()
    num_iter = args.num_iter


    # TODO implement schedule generation
    schedule = []
    if args.file == "tiny_pangenome.og":
        print("Using hardcoded schedule of tiny_pangenome.og")
        # for tiny_pangenome
        schedule.append(100)
        schedule.append(72.7895)
        schedule.append(52.9832)
        schedule.append(38.5662)
        schedule.append(28.0722)
        schedule.append(20.4336)
        schedule.append(14.8735)
        schedule.append(10.8264)
        schedule.append(7.88046)
        schedule.append(5.73615)
        schedule.append(4.17532)
        schedule.append(3.0392)
        schedule.append(2.21222)
        schedule.append(1.61026)
        schedule.append(1.1721)
        schedule.append(0.853168)
        schedule.append(0.621017)
        schedule.append(0.452035)
        schedule.append(0.329034)
        schedule.append(0.239503)
        schedule.append(0.174333)
        schedule.append(0.126896)
        schedule.append(0.0923671)
        schedule.append(0.0672336)
        schedule.append(0.048939)
        schedule.append(0.0356225)
        schedule.append(0.0259294)
        schedule.append(0.0188739)
        schedule.append(0.0137382)
        schedule.append(0.01)
        schedule.append(0.00727895)

    elif args.file == "DRB1-3123.og":
        print("Using hardcoded schedule of DRB1-3123.og")
        # for DRB1-3123
        schedule.append(.61e+06)
        schedule.append(4.70949e+06)
        schedule.append(2.30794e+06)
        schedule.append(1.13104e+06)
        schedule.append(554277)
        schedule.append(271630)
        schedule.append(133116)
        schedule.append(65234.9)
        schedule.append(31969.1)
        schedule.append(15666.8)
        schedule.append(7677.73)
        schedule.append(3762.56)
        schedule.append(1843.89)
        schedule.append(903.619)
        schedule.append(442.829)
        schedule.append(217.014)
        schedule.append(106.35)
        schedule.append(52.1181)
        schedule.append(25.5411)
        schedule.append(12.5167)
        schedule.append(6.13397)
        schedule.append(3.00603)
        schedule.append(1.47314)
        schedule.append(0.721929)
        schedule.append(0.35379)
        schedule.append(0.173379)
        schedule.append(0.0849664)
        schedule.append(0.0416388)
        schedule.append(0.0204056)
        schedule.append(0.01)
        schedule.append(0.00490062)

    else:
        sys.exit("ERROR: Unable to find hardcoded schedule for file name")


    if args.num_iter > len(schedule):
        sys.exit("ERROR: Hardcoded schedule only available for {} iterations".format(len(schedule)))


    x = torch.rand([n,2,2], dtype=torch.float64, requires_grad=True)

    start = datetime.now()
    # ***** Interesting NOTE: I think odgi runs one iteration more than selected with argument
    for iteration, eta in enumerate(schedule[:num_iter]):
        print("Computing iteration", iteration + 1, "of", num_iter, eta)
        
        for batch_idx, (i, j, vis_p_i, vis_p_j, _w, dis) in enumerate(data):
            # breakpoint()

            # pytorch model as close as possible to odgi implementation

            # compute weight w in PyTorch model (here); don't use computed weight of dataloader
            # dataloader computes it as w = 1 / dis^2
            # ***** Interesting NOTE (found by Jiajie): ODGI uses as weight 1/dis; while original graph drawing paper uses 1/dis^2
            
            # batch_sgd.py
            '''
            w = 1 / dis
            mu = eta * w
            mu_m = torch.min(mu, torch.ones_like(mu))

            lr = torch.min(mu_m / (4*w))

            stress = q(x[i-1,vis_p_i], x[j-1,vis_p_j], dis)
            stress.backward()
            x.data.sub_(lr * x.grad.data) # lr set to be a vector?
            x.grad.data.zero_()
            '''

            # batch_sgd_vector_lr.py
            w = 1 / dis
            mu = eta * w
            mu_m = torch.min(mu, torch.ones_like(mu))

            lr = torch.zeros((x.shape[0], x.shape[1], 1))
            lr_value = mu_m / (4 * w) # vector
            batch_pair = list(zip(i,j,vis_p_i,vis_p_j))
            
            for pair_idx, pair in enumerate(batch_pair): # we give each batch a different LR -> consistent with original paper
                lr[pair[0]-1,pair[2]] += lr_value[pair_idx]
                lr[pair[1]-1,pair[3]] += lr_value[pair_idx]

            stress = q(x[i-1,vis_p_i], x[j-1,vis_p_j], dis)
            stress.backward()
            x.data.sub_(lr * x.grad.data) # lr set to be a vector?
            x.grad.data.zero_()

            if (torch.isnan(x).any()):
                raise ValueError(f"Iter[{iteration}] Step[{batch_idx}]. x: {x} is NaN. lr: {lr}. w: {w}.")

        if ((iteration+1) % 5) == 0 and args.create_iteration_figs == True:
            x_np = x.clone().detach().numpy()
            draw_svg(x_np, data, "output/out_iter{}".format(iteration+1))

    end = datetime.now()
    print("Computation took", end-start)

    x_np = x.detach().numpy()
    OdgiInterface.generate_layout_file(data.get_graph(), x_np, "output/" + args.file + ".lay")
    draw_svg(x_np, data, "output/out_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SGD Implementation for Graph Drawing")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--file', type=str, default='tiny_pangenome.og', help='odgi variation graph')
    parser.add_argument('--create_iteration_figs', action='store_true', help='create at each 5 iteration a figure of the current state')
    args = parser.parse_args()
    print(args)
    main(args)