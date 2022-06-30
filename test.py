import torch
import torch.nn as nn
import numpy as np

# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

#     def forward(self, x):
#         # ParameterList can act as an iterable, or be indexed using ints
#         for i, p in enumerate(self.params):
#             x = self.params[i // 2].mm(x) + p.mm(x)
#         return x

# mod = MyModule()
# # print(mod.parameters())
# # print(mod.params)

# num_nodes = 882
# pos = np.random.rand(num_nodes, 2)
# param = nn.Parameter(torch.from_numpy(pos))
# print(param)

# a = torch.tensor([10.0, 10.0], requires_grad=True)
# b = torch.tensor([20.0, 20.0], requires_grad=True)

# F = a * b
# print(F)
# F.backward(gradient=torch.tensor([2.0, 1.0]))

# print(a.grad)
# print(b.grad)


num_nodes = 882
dis = np.random.randn(num_nodes, num_nodes)
breakpoint()
dis_torch = torch.from_numpy(dis)

