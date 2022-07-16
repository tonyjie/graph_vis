# Graph Visualization
The goal is to get the visualization of the graph.
- Input: the graph G = (V, E). We can get the *real distance* between each node (here is the shortest distance in the graph). Then we want our visualization graph to be as close as possible to the *real distance* guidance. 
- Output: each node is assigned a coordinate (X, Y). 


## Code
- `graphdraw.py`
    - [the original paper's algorithm](https://github.com/jxz12/s_gd2/blob/master/jupyter/main.ipynb). It is just simple math without Pytorch implementation. It is not involved with gradient computing.
- `torch_sgd.py`
    - Implement the algorithm in Pytorch. It is fixed to be BATCH_SIZE = 1. The computation should be the same with the original algorithm. The stress function is regarded as the loss function, and `.backward()` is called to get the gradient for each term. Each time we update a pair of nodes. 
- `batch_sgd.py`
    - Pytorch implementation that supports any BATCH_SIZE. Note that the original algorithm is designed only for BATCH_SIZE = 1, and has some specific constraints. To make it reasonable to apply to BATCH_SIZE > 1, I made some modifications, mainly on the learning_rate. When BATCH_SIZE = 1, it should be the same sas `torch_sgd.py` and `graphdraw.py`. 
    - Very slow. 
- `batch_sgd_vector_lr.py`
    - Pytorch implementation that supports any BATCH_SIZE. The learning rate is set to be a vector, that aims to be consistent with the original algorithm setting, because each pair of nodes have different learning rate based on their distance. 
        ```python
        wc = w * c
        wc = torch.min(wc, torch.ones_like(wc))
        lr = torch.min(wc / (4 * w))
        x.data.sub_(lr * x.grad.data)
        ```
    - Very slow. 
- `full_graph_sgd.py`
    - Full Graph update. It just compute the overall stress function, and compute the gradient and update. It partly works on the general graph (some SuitSparse testcases): can generate reasonable viz graph, but the stress is larger than the optimal one. 
    - Super fast. 
    - Can try different optimizers (Adam, SGD) and learning rate scheduler. The code didn't follow the original algorithm. It didn't have a `cap` operation to regulate the learning rate for each pairwise currently. 
- `odgi_dataset.py`, `odgi_batch_sgd.py`
    - Extend `batch_sgd.py` on ODGI dataloader to support real pangenome input. For BATCH_SIZE = 1, it can generate recognizable results for DRB1-3123. 
    - Slow. ~40 minutes for DRB1-3123. 
- `odgi_full_graph_sgd.py`
    - Full Graph update. Extend `full_graph_sgd.py` on pangenome input. Use ODGI's Python API to compute the Distance Matrix for each paths. But the algorithm doesn't converge currently, though the training process is fast. 








# Experiments for `batch_sgd.py` (All on local CPU)
We would evaluate on two aspects:
- Time consumption & Iteration times
- Quality: graph visualization qualitatively; stress quantitatively. 

## Original Graph Drawing implementation.
### Iter_num = 15
- Time: 50.07s
- Result: `output/qh882.svg`; Stress = 18736

## Batch SGD. 
### Batch Size = 1. Iter_num = 15
- Time: 18min 46s
- Result: `output/batch_1_iter_15.svg`; Stress = 18741

### Batch Size = 16. Iter_num = 15
- Time: 1 min 20s
- Result: `output/batch_16_iter_15.svg`; Stress = 45287

### Batch Size = 16. Iter_num = 60
- Time: 5min 10s
- Result: `output/batch_16_iter_60.svg`; Stress = 18759

### Batch Size = 256. Iter_num = 15
- Time: 12s
- Stress = 237022

### Batch Size = 256. Iter_num = 60
- Time: 44s
- Stress = 187087

### Batch Size = 256. Iter_num = 240
- Time: 3min 7s
- Stress = 126226

## Batch SGD with Vector Learning Rate
### Batch Size = 1. Iter_num = 15. 
- Time: 23min 0s
- Stress: 18732

### Batch Size = 16. Iter_num = 15. 
- Time: 3min 13s
- Stress: 30224

### Batch Size = 16. Iter_num = 30
- Time: 6min 10s
- Stress: 18841

### Batch Size = 256. Iter_num = 15. 
- Time: 2min 26s
- Stress: 55679

# Experiments `batch_sgd.py` (Server: zhang-capra-xcel)
## Setup
### Hardware
CPU: Intel Xeon Gold 6248R CPU @ 3.00GHz, 48 Cores. 
GPU: 4 * Nvidia RTX A6000, 48GB

## 1. Dataset: qh882 (num_nodes = 882)
qh882. Num_nodes = 882.     

###  Original Graph Drawing Python Implementation.
Iter_num = 15. 
- Time: 57.7s
- Stress: 18737

###  Batch SGD.
Initial Stress = 349943

| Batch Size | Iter_num | Time (CPU) | Stress (CPU) | Time (GPU) | Stress (GPU) | 
| ---------- | -------- | ---------- | ------------ | ---------- | ------------ |
| 1          | 15       | too long   |              | too long   | -            | 
| 4          | 15       | 9min 7s    | 18758        | 17min 38s  | 18797        |
| 16         | 15       | 2min 45s   | 27883        | 4min 44s   | 34499        |
| 64         | 15       | 1min 4s    | 171877       | 1min 32s   | 175903       |
| 256        | 15       | 39s        | 237467       | 45s        | 237673       |
| 1024       | 15       | 32.7s      | 323677       | 28.2s      | 324485       |

### Batch SGD with Vector Learning Rate. Dataset = qh882 (n=882)

| Batch Size | Iter_num | Time (CPU) | Stress (CPU) | Time (GPU) | Stress (GPU) | 
| ---------- | -------- | ---------- | ------------ | ---------- | ------------ |
| 1          | 15       | too long   |              | too long   | -            | 
| 4          | 15       | 9min 23s   | 18792        | 33min 25s  | 18739        |
| 16         | 15       | 3min 55s   | 24297        | 8min 18s   | 19037        |
| 64         | 15       | 2min 32s   | 38572        | 4min 41s   | 40610        |
| 256        | 15       | 2min 13s   | 39355        | 2min 56s   | 31199        |
| 1024       | 15       | 2min 18s   | 53263        | 2min 55s   | 61791        |

## 2. Dataset: bcspwr10. Num_nodes = 5300.    
###  Original Graph Drawing Python Implementation.
Iter_num = 15. 
- Time: 35min 48s
- Stress: 679064

### Batch SGD.

Initial Stress = 13,213,179

| Batch Size | Iter_num | Time (CPU) | Stress (CPU) | Time (GPU) | Stress (GPU) | 
| ---------- | -------- | ---------- | ------------ | ---------- | ------------ |
| 1          | 15       | too long   |              | too long   | -            | 
| 4          | 15       |     | xxxx        |   | xxxx        |
| 16         | 15       | 102min 35s | 679,289      | 177min 13s | 679,301      |
| 64         | 15       | 37min 33s  | 2,178,907    |  55min 28s | 1,077,146    |
| 256        | 15       | 19min 05s  | 6,196,725    | 25min 11   | 6,159,278    |
| 1024 (weird but true)      | 15       | 46min 59s  |   9,033,926    | 48min 22s  | 9,031,573      |





