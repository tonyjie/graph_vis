# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# Run with command 'env LD_PRELOAD=libjemalloc.so.2 PYTHONPATH=<lib dir of odgi-build> python3 batch_sgd.py --batch_size 100 --num_iter 30'

import argparse
import sys
import math
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


def main(args):
    start = datetime.now()
    data = OdgiDataloader(args.file, batch_size=args.batch_size)

    n = data.get_node_count()
    num_iter = args.num_iter


    w_min = data.w_min
    w_max = data.w_max
    epsilon = 0.01              # default value of odgi

    eta_max = 1/w_min
    eta_min = epsilon/w_max
    lambd = math.log(eta_min / eta_max) / (num_iter - 1)
    schedule = []
    for t in range(num_iter):
        eta = eta_max * math.exp(lambd * t)
        schedule.append(eta)


    x = torch.rand([n,2,2], dtype=torch.float64)

    # ***** Interesting NOTE: I think odgi runs one iteration more than selected with argument
    for iteration, eta in enumerate(schedule[:num_iter]):
        print("Computing iteration", iteration + 1, "of", num_iter, eta)
        
        for batch_idx, (i, j, vis_p_i, vis_p_j, _w, dis) in enumerate(data):
            # print("batch", batch_idx, "of", math.floor(float(data.steps_in_iteration()) / float(data.batch_size)))
            # pytorch model as close as possible to odgi implementation

            # compute weight w in PyTorch model (here); don't use computed weight of dataloader
            # dataloader computes it as w = 1 / dis^2
            # ***** Interesting NOTE (found by Jiajie): ODGI uses as weight 1/dis; while original graph drawing paper uses 1/dis^2
            w = 1 / dis

            mu = eta * w
            mu_m = torch.min(mu, torch.ones_like(mu))

            x_i = x[i-1,vis_p_i,0]
            x_j = x[j-1,vis_p_j,0]
            y_i = x[i-1,vis_p_i,1]
            y_j = x[j-1,vis_p_j,1]

            dx = x_i - x_j
            dy = y_i - y_j

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
