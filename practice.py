import torch


letter_indexes = ["abcd".find("acb"[li]) for li in range(1, len("acb"))] # [2, 1]

a = torch.LongTensor(letter_indexes)
print(a) # tensor([1, 2, 3])
# print(a.unsqueeze_(0)) # tensor([[1, 2, 3]])
# print(a.unsqueeze_(1)) # tensor([[1],
                                 # [2],
#                                  [3]])
print(a.unsqueeze_(-1))
