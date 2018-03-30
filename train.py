# https://github.com/hiepph/char-rnn

import argparse
import codecs
import string
import random
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import RNN


# Data
all_characters = string.printable
n_characters = len(all_characters)
chunk_len = 200

# Hyper parameters
n_hidden = 50
n_layers = 2


def char_tensor(string, gpu=-1):
    """Turn string into list of longs"""
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c]  = all_characters.index(string[c])

    if gpu >= 0:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

def random_chunk(source):
    start_index = random.randint(0, len(source) - chunk_len)
    end_index = start_index + chunk_len + 1
    return source[start_index:end_index]

def random_training_set(source, gpu=-1):
    chunk = random_chunk(source)
    inp = char_tensor(chunk[:-1], gpu)
    target = char_tensor(chunk[1:], gpu)

    return inp, target


class Model():
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, gpu=-1):
        self.decoder = RNN(input_size, hidden_size, output_size, n_layers, gpu)
        if gpu >= 0:
            print("Use GPU %d" % torch.cuda.current_device())
            self.decoder.cuda()

        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, inp, target, chunk_len=200):
        hidden = self.decoder.init_hidden()
        self.decoder.zero_grad()
        loss = 0

        for c in range(chunk_len):
            out, hidden = self.decoder(inp[c], hidden)
            loss += self.criterion(out, target[c])

        loss.backward()
        self.optimizer.step()

        return loss.data[0] / chunk_len

    def generate(self, prime_str, predict_len=100, temperature=0.8):
        predicted = prime_str

        hidden = self.decoder.init_hidden()
        prime_input = char_tensor(prime_str, self.decoder.gpu)

        # Use prime string to build up hidden state
        for p in range(len(prime_str) - 1):
            _, hidden = self.decoder(prime_input[p], hidden)

        inp  = prime_input[-1]
        for p in range(predict_len):
            out, hidden = self.decoder(inp, hidden)

            # sample from network as a multinomial distribution out_dist = out.data.view(-1).div(temperature).exp()
            out_dist = out.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(out_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_char = all_characters[top_i]
            predicted += predicted_char
            inp = char_tensor(predicted_char, self.decoder.gpu)

        return predicted

    def save(self):
        model_name = "char-rnn-gru.pt"

        if not os.path.exists("save"):
            os.mkdir("save")
        torch.save(self.decoder, "save/%s" % model_name)
        print("--------------> [Checkpoint] Save model into save/%s" % model_name)

    def load(self, model_path="save/char-rnn-gru.pt"):
        self.decoder = torch.load(model_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--source', type=str, required=True, help='Training text file')
    argparser.add_argument('--epoch', type=int, default=2000, help='Number of epochs')
    argparser.add_argument('--frequency', type=int, default=50, help='Frequently check loss with interval, and save model')
    argparser.add_argument('--gpu', type=int, default=-1, help='Id of GPU (-1 indicates CPU)')
    args = argparser.parse_args()

    source = codecs.open(args.source, "r", encoding="utf-8", errors="ignore").read()

    # Build model
    model = Model(n_characters, n_hidden, n_characters, n_layers, args.gpu)

    # Train
    print("Train with %d epochs" % args.epoch)

    for e in range(args.epoch):
        while True:
            try:
                ts = random_training_set(source, args.gpu)
            except ValueError:
                print "bad set"
            else:
                break

        loss = model.train(*ts)

        if (e+1) % args.frequency == 0:
            print("\n--------> [EPOCH %d] loss %.4f" % (e+1, loss))
            prime_sample = random.choice(string.ascii_letters)
            print(model.generate(prime_sample))

            model.save()
