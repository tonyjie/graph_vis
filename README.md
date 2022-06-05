# Graph Visualization

## Code
- `python graphdraw.py`    
It would run the original paper's algorithm. It is just simple math without Pytorch implementation. It is not involved with gradient computing. 

- `python torch_sgd.py`     
Implement the algorithm in Pytorch. It is fixed to be BATCH_SIZE = 1. The computation should be the same with the original algorithm. The stress function is regarded as the loss function, and `.backward()` is called to get the gradient for each term. Each time we update a pair of nodes. 

- `python batch_sgd.py`     
Pytorch implementation that supports any BATCH_SIZE. Note that the original algorithm is designed only for BATCH_SIZE = 1, and has some specific constraints. To make it reasonable to apply to BATCH_SIZE > 1, I made some modifications, mainly on the learning_rate. 

```python
wc = w * c
wc = torch.min(wc, torch.ones_like(wc))
lr = torch.min(wc / (4 * w))
x.data.sub_(lr * x.grad.data)
```

## Experiments (All on local CPU)
We would evaluate on two aspects:
- Time consumption & Iteration times
- Quality: graph visualization qualitatively; stress quantitatively. 

### Original Graph Drawing implementation. Iter_num = 15
- Time: 50.07s
- Result: `qh882.svg`; Stress = 18736

### Batch SGD. Batch Size = 1. Iter_num = 15
- Time: very long
- Result: `batch_1_iter_15.svg`; Stress = 18741


### Batch SGD. Batch Size = 16. Iter_num = 15
- Time: 1 min 20s
- Result: `batch_16_iter_15.svg`; Stress = 45287

### Batch SGD. Batch Size = 16. Iter_num = 60
- Time: 5min 10s
- Result: `batch_16_iter_60.svg`; Stress = 18759

