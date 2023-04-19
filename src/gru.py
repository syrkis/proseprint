# gru.py
#     gru model for mawsa
# by: Noah Syrkis

# imports
from src.train import train
from src.utils import UNK, UNK_IDX, PAD, encode, decode, flatten, make_char_vocab, make_word_vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np


# gru
class GRUSiam(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(GRUSiam, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)

    def forward(self, x, y):
        x = self.word_embeddings(x)
        x = self.gru(x)
        loss = F.nll_loss(x, y)
        return x, loss

def gru_siam_batch_fn(data, block_size, batch_size):
    idxs = np.random.choice(len(data), batch_size)


# functions
def run_gru_siam(train_data, valid_data, conf):
    stoi, itos = make_word_vocab(train_data['text'].values) 
    conf.vocab_size = len(stoi)
    conf.vocab = stoi
    train_data['idxs'] = train_data['text'].apply(lambda x: encode(x, conf.vocab))
    valid_data['idxs'] = valid_data['text'].apply(lambda x: encode(x, conf.vocab))
    model = GRUSiam(conf.embedding_dim, conf.hidden_dim, conf.vocab_size, conf.tagset_size)
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    train = train(model, train_data, valid_data, optimizer, conf, gru_siam_batch_fn)
