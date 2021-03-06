# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# we’ll train on a few thousand surnames from 18 languages of origin, and predict which language a name is from based on the spelling


"""
        (discard)        (discard)           (final output)
        output1           output2               outputN
hidden     |     hidden     |           hidden     |       (don't care)
------>[      ]--------->[      ]-----.....---->[      ]---------------->
           |                |                      |
        letter1          letter2                letterN
"""

# Todo
# Add more linear layers
# Try the nn.LSTM and nn.GRU layers
# Combine multiple of these RNNs as a higher level network

from io import open
import glob
import os
import torch
import torch.nn as nn
import unicodedata
import string
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

print("ASCII for Ślusàrski:", unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {} # (language: [name1, name2, ...]) dictionary
all_categories = [] # list of languages

# Read a file and return a list with each item being Ascii version of the a line in the file
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('../data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print("all_categories: ", all_categories)
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
    def __init__(self, input_size, hidden_size, output_size): # each char is of 57 long one-hot vector (input_size), output_size is n_categories for softmax
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # hidden layer of size (input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # output layer of size (input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden): # input [1, 57]; hidden [1, 128]
        combined = torch.cat((input, hidden), 1) # Concatenates the given tensors (all tensors have the same shape) in the given direction, combined [1, 185]
        hidden = self.i2h(combined) # hidden [1, 128]
        output = self.i2o(combined) # output [1, 18]
        output = self.softmax(output) # output [1, 18]
        return output, hidden

    def initHidden(self): # we initialize as zeros for the first hidden layer
        return torch.zeros(1, self.hidden_size)

n_hidden = 128 # define hidden_size
rnn = RNN(n_letters, n_hidden, n_categories)  # instantiate RNN we built

input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input, hidden) # same input as forward()

input = lineToTensor('Albert') # For the sake of efficiency we don’t want to be creating a new Tensor for every step
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
print("RNN first layer output for A:", output) # output is a <1 x n_categories> Tensor, where every item is the likelihood of that category (higher is more likely).

# translate softmax output to category
def categoryFromOutput(output):
    top_n, top_i = output.topk(1) # Returns the k largest elements and index
    category_i = top_i[0].item() # index
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories) # a random language
    line = randomChoice(category_lines[category]) # a random name from that language
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long) # index of that language in form of a tensor
    line_tensor = lineToTensor(line) # a <line_length x 1 x n_letters> array of one-hot letter vectors
    return category, line, category_tensor, line_tensor

# for i in range(2):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ line =', line)
#     print(category_tensor)

criterion = nn.NLLLoss() # The negative log likelihood loss. It is useful to train a classification problem with C classes.

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor): # category_tensor [1], line_tensor [nameLength, 1, 57]
    hidden = rnn.initHidden()
    rnn.zero_grad() # Sets gradients of all model parameters to zero.
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden) # discard intermediate output, feed previous hidden to the next layer

    loss = criterion(output, category_tensor) # output is the last output, output [1, 18] category_tensor [1]
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters(): # Returns an iterator over module parameters.
        p.data.add_(-learning_rate, p.grad.data)
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since): # return (x mins y secs) since a given time point
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss # total loss for each plot iteration

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every) # plot avg loss for each plot iteration
        current_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()


# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')


model_path = './model.txt'
# torch.save(rnn, model_path)
# rnn = torch.load(model_path)
