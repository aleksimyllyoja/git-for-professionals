# https://github.com/hiepph/char-rnn

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, gpu=-1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.gpu = gpu

    def forward(self, inp, hidden):
        inp = self.encoder(inp.view(1, -1))
        out, hidden = self.gru(inp.view(1, 1, -1), hidden)
        out = self.decoder(out.view(1, -1))

        return out, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        if self.gpu >= 0:
            return Variable(hidden.cuda())
        else:
            return Variable(hidden)
