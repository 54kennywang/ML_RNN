import torch

a = torch.LongTensor([1, 2, 3])
print(a) # tensor([1, 2, 3])
# print(a.unsqueeze_(0)) # tensor([[1, 2, 3]])
# print(a.unsqueeze_(1)) # tensor([[1],
                                 # [2],
#                                  [3]])
print(a.unsqueeze_(-1))
