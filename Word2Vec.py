from Tokenizer import Tokenizer as tk
from FullyConnected import FullyConnected as fc
from torch import nn
import torch
import gensim
import warnings
warnings.filterwarnings(action='ignore')


def main():
    tok = tk('./Dataset/lyrics15LIN.csv', ['english', 'spanish'])

    model = gensim.models.Word2Vec(tok.tokenize(), min_count=20, size=300, window=5, sg=1)
    net = fc()
    # create a stochastic gradient descent optimizer
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    print(model.similarity('man', 'something'))
    b = (model['king'] - model['man'] + model['woman'])
    print(model.similar_by_vector(b))
    b = (model['jesus'] + model['cross'])
    print(model.similar_by_vector(b))


if __name__ == "__main__":
    main()