import torch


t = torch.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True, dtype=torch.float32)
y = t.sum()
y.backward()
print(t.grad)
