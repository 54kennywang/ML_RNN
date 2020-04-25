# instead of predicting a category after inputting a name, we input a category and output one letter at a time.
# and recurrently predicting characters to form language in that category

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('../data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))

# The category tensor is a one-hot vector just like the letter input
# interpret the output as the probability of the next letter. When sampling, the most likely output letter is used as the next input letter.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size) # for hidden out for next layer
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size) # output of current layer
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1) # [category]+[input]+[hidden]=[category, input, hidden]
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

hidden_size = 128
rnn = RNN(n_letters, hidden_size, n_letters)

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# For each timestep (that is, for each letter in a training word) the inputs of the network will be (category, current letter, hidden state)
# and the outputs will be (next letter, next hidden state). So for each training set, we’ll need the category, a set of input letters, and a set of output/target letters.
# Since we are predicting the next letter from the current letter for each timestep, the letter pairs are groups of consecutive letters from the line
# - e.g. for "ABCD<EOS>" we would create (“A”, “B”), (“B”, “C”), (“C”, “D”), (“D”, “EOS”).

# One-hot vector for category, shape = (1, n_categories=18)
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of of a name (not including EOS) for input, shape = (name_length, 1, n_letters=58)
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target, shape = (name_length, 1, n_letters=58), same as inputTensor (+1 for EOS, -1 for the first letter)
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))] # list of index of the letters (from second one) in the given name. "acb" = [2, 1]
    letter_indexes.append(n_letters - 1) # EOS because n_letters = len(all_letters) + 1 # Plus EOS marker
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

criterion = nn.NLLLoss()
learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)


"""
        letter2_pred-------           letter3_pred----------                    letterN_pred
hidden        |           |   hidden        |              |     hidden                |              (don't care)
------>[            ]-----|--------->[            ]--------|-------.....---->[                    ]---------------->
            /  \          |               /  \             |                          / \
           /    \         |              /    \            |                         /   \
      category  letter1   ------>  category  letter2_pred  ---------...-----> category  letter(n-1)_pred

Given word "Mike" and English, we feed in "M", RNN predict xyz, hopefully, xyz="ike".
"""

