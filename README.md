# Graph Visualization
The goal is to get the visualization of the graph.
- Input: the graph G = (V, E). We can get the *real distance* between each node (here is the shortest distance in the graph). Then we want our visualization graph to be as close as possible to the *real distance* guidance. 
- Output: each node is assigned a coordinate (X, Y). 


## Code
- `python graphdraw.py`    
It would run [the original paper's algorithm](https://github.com/jxz12/s_gd2/blob/master/jupyter/main.ipynb). It is just simple math without Pytorch implementation. It is not involved with gradient computing. 

- `python torch_sgd.py`     
Implement the algorithm in Pytorch. It is fixed to be BATCH_SIZE = 1. The computation should be the same with the original algorithm. The stress function is regarded as the loss function, and `.backward()` is called to get the gradient for each term. Each time we update a pair of nodes. 

- `python batch_sgd.py`     
Pytorch implementation that supports any BATCH_SIZE. Note that the original algorithm is designed only for BATCH_SIZE = 1, and has some specific constraints. To make it reasonable to apply to BATCH_SIZE > 1, I made some modifications, mainly on the learning_rate. 

- `python batch_sgd_vector_lr.py`       
Pytorch implementation that supports any BATCH_SIZE. The learning rate is set to be a vector, that aims to be consistent with the original BATCH_SIZE = 1 setting. (But there might be some bugs. )

```python
wc = w * c
wc = torch.min(wc, torch.ones_like(wc))
lr = torch.min(wc / (4 * w))
x.data.sub_(lr * x.grad.data)
```

# Experiments (All on local CPU)
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
### Batch Size = 16. Iter_num = 15. 
- Time: 3min 13s
- Stress: 30224

### Batch Size = 16. Iter_num = 30
- Time: 6min 10s
- Stress: 18841