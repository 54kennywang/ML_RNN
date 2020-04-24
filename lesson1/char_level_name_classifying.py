# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# we’ll train on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling
from io import open
import glob
import os
import torch
import torch.nn as nn
import unicodedata
import string

def findFiles(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {} # (language: [name1, name2, ...]) dictionary
all_categories = [] # list of languages

# Read a file and return a list with each item being Ascii version of the a line in the file
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a letter into a <1 x n_letters> Tensor, e.g. "a" = [[1, 0, 0, ...]]
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters> array of one-hot letter vectors.
# That extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))
print(lineToTensor('Jones').size()) # [5, 1, 57], 57 = 26*2+5 " .,;'"


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): # each char is of 57 long one-hot vector (output_size), output_size is n_categories for softmax
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # hidden layer of size (input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # output layer of size (input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) # Concatenates the given tensors (all tensors have the same shape) in the given direction
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self): # we initialize as zeros for the first hidden layer
        return torch.zeros(1, self.hidden_size)

n_hidden = 128 # define hidden_size
rnn = RNN(n_letters, n_hidden, n_categories)  # instantiate RNN we built

input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden)
output, next_hidden = rnn(input, hidden) # same input as forward()


input = lineToTensor('Albert') # For the sake of efficiency we don’t want to be creating a new Tensor for every step
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
print(output) # output is a <1 x n_categories> Tensor, where every item is the likelihood of that category (higher is more likely).

# continue with training https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#training

