import torch
import torch.nn as nn
import torch.nn.functional as F

embedding = nn.Embedding(10, 7) # (inputSize, hiddenSize)
gru = nn.GRU(7, 7)

hidden = torch.zeros(1, 1, 7)
print(hidden) # (1, 1, 7)
words = [3, 5, 2]
x = torch.tensor(words, dtype=torch.long).view(-1, 1)
print(x) # (1, 3)
print(x[0]) # (1)
embedded_input = embedding(x[0]).view(1, 1, -1)
print(embedded_input.shape) # [1, 1, 7]
x1 = torch.tensor([[0]])
embedded_input1 = embedding(x1).view(1, 1, -1)
print("haha", embedded_input.shape) # [1, 1, 7]
print(embedded_input) # [1, 1, 7]
output, hidden = gru(embedded_input, hidden)
print(output) # [1, 1, 7]
print(hidden) # [1, 1, 7]

print(torch.tensor([[0]])) # (1, 1)

attn = nn.Linear(7 * 2, 11) # (hiddenSize*2, outputSize)
m = torch.cat((embedded_input[0], hidden[0]), 1)
print("yes", m.shape) # [1, 14]
n = attn(m)
print(n.shape) # [1, 11]
k = F.softmax(n, dim=1)
print(k.shape) # [1, 11]
t = k.unsqueeze(0)
print(t.shape) # [1, 1, 11]
s = torch.zeros(11, 7).unsqueeze(0)
print(s.shape) # [1, 11, 7]
a = torch.bmm(t, s) # [1, 1, 11] * [1, 11, 7]
print(a.shape) # [1, 1, 7]


z = torch.zeros(1, 1, 7)
print(F.relu(z).shape) # [1, 1, 7]

v = nn.GRU(7, 7)
o, h = v(z, z)
print(o.shape) # [1, 1, 7]
print(h.shape) # [1, 1, 7]

out = nn.Linear(7, 11)
x = out(o[0])
print(x.shape)# (1, 11)
print((F.log_softmax(x, dim=1)).shape) # (1, 11)