import torch

a = torch.Tensor([[1,2,3,4,5], [5,6,7,8,9]])
b = torch.randn((5, 1))
c = a @ b
print(c.shape)


print("hello")