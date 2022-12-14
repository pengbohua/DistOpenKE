import torch

a = torch.tensor([[1,2,3,4], [5,6,7,8]])
b = torch.split(a, 2, dim=-1)
print(b[0])