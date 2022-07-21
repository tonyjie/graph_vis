# Use with odgi branch zhang_research_extended https://github.com/nsmlzl/odgi/tree/zhang_research_extended
# Run with command 'env LD_PRELOAD=libjemalloc.so.2 PYTHONPATH=<lib dir of odgi-build> python3 batch_sgd.py --batch_size 1 --num_iter 15'
# env LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so PYTHONPATH=~/odgi_niklas/lib python odgi_model.py --file DRB1-3123.og --batch_size=100 --num_iter=31 --create_iteration_figs
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
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"==== Device: {device}; Dataset: {args.file} ====")

    data = OdgiDataloader(args.file)
    data.set_batch_size(args.batch_size)

    n = data.get_node_count()
    num_iter = args.num_iter

    w_min = data.w_min
    w_max = data.w_max
    epsilon = 0.01              # default value of odgi

    eta_max = 1/w_min
    eta_min = epsilon/w_max
    lambd = math.log(eta_min / eta_max) / (num_iter - 1)
    schedule_c = []
    for t in range(num_iter):
        eta = eta_max * math.exp(lambd * t)
        schedule_c.append(eta)



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
        schedule.append(9.61e+06)
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

    print(f"Compare schedule: {schedule==schedule_c}")
    print(f"Schedule: {schedule}")
    print(f"Schedule_c: {schedule_c}")
    sys.exit()


    print(f"len(data): {data.steps_in_iteration()}") # 350590
    # torch.set_num_interop_threads(1)
    # print(f"==== Config: num_threads: {torch.get_num_threads()}; num_interop_threads: {torch.get_num_interop_threads()} ====")

    x = torch.rand([n,2,2], dtype=torch.float64, device=device)

    start = datetime.now()
    
    compute_time = 0

    # ***** Interesting NOTE: I think odgi runs one iteration more than selected with argument
    for iteration, eta in enumerate(schedule[:num_iter]):
        print("Computing iteration", iteration + 1, "of", num_iter, eta)
        
        for batch_idx, (i, j, vis_p_i, vis_p_j, _w, dis) in enumerate(data):
            # breakpoint()

            # pytorch model as close as possible to odgi implementation

            # compute weight w in PyTorch model (here); don't use computed weight of dataloader
            # dataloader computes it as w = 1 / dis^2
            # ***** Interesting NOTE (found by Jiajie): ODGI uses as weight 1/dis; while original graph drawing paper uses 1/dis^2
            
            compute_start = datetime.now()

            # to cuda
            dis = dis.to(device)

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

            delta = mu_m * (mag - dis) / 2.0

            r = delta / mag
            r_x = r * dx
            r_y = r * dy

            x[i-1, vis_p_i, 0] = x[i-1, vis_p_i, 0] - r_x
            x[j-1, vis_p_j, 0] = x[j-1, vis_p_j, 0] + r_x
            x[i-1, vis_p_i, 1] = x[i-1, vis_p_i, 1] - r_y
            x[j-1, vis_p_j, 1] = x[j-1, vis_p_j, 1] + r_y

            compute_end = datetime.now()
            compute_time += (compute_end - compute_start).total_seconds()


        if ((iteration+1) % 5) == 0 and args.create_iteration_figs == True:
            x_np = x.cpu().clone().detach().numpy()
            draw_svg(x_np, data, "output/out_iter{}".format(iteration+1))

    end = datetime.now()
    overall_time = (end-start).total_seconds()
    dataload_time = overall_time - compute_time
    print(f"Overall time {overall_time} sec; Dataloading time: {dataload_time} sec; Computation took {compute_time} sec")

    x_np = x.cpu().detach().numpy()
    OdgiInterface.generate_layout_file(data.get_graph(), x_np, "output/" + args.file + ".lay")
    draw_svg(x_np, data, "output/out_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SGD Implementation for Graph Drawing")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=15, help='number of iterations')
    parser.add_argument('--file', type=str, default='tiny_pangenome.og', help='odgi variation graph')
    parser.add_argument('--create_iteration_figs', action='store_true', help='create at each 5 iteration a figure of the current state')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    args = parser.parse_args()
    print(args)
    main(args)