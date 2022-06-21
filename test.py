import enum
import torch
from torch.utils.data import TensorDataset, DataLoader

# inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
# tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
# dataset = TensorDataset(inps, tgts)

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# breakpoint()

# for batch_ndx, (inps, tgts) in enumerate(dataloader):
#     print(f"batch_ndx: {batch_ndx}, inps: {inps}, tgts: {tgts}")

breakpoint()
a = torch.Tensor([1,2,3,-2])
b = torch.ones_like(a)
c = torch.max(a, b)
print(c)