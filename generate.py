# https://github.com/hiepph/char-rnn

import argparse
import warnings
import random

from train import Model
from train import n_characters, n_hidden, n_layers

def straight_flames():
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)));

    model = Model(n_characters, n_hidden, n_characters, n_layers)
    model.load("models/wtchrrnn.pt")

    return " ".join(model.generate("Add ", 40).split(" ")[:-1]).strip()+"."

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default="save/char-rnn-gru.pt", help="Path to trained model")
    argparser.add_argument('--prime', type=str, required=True, help="Prime string to predict next sequence of characters")
    argparser.add_argument('--len', type=int, default=1000, help="Predict string length")
    args = argparser.parse_args()

    warnings.filterwarnings("ignore")

    model = Model(n_characters, n_hidden, n_characters, n_layers)
    model.load(args.model)

    print(model.generate(args.prime, args.len))
