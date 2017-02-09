from functools import partial
import argparse
import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import BaseSentencePairTrainer, Embed, to_gpu
from spinn.util.misc import Args, Vocab


class SentencePairTrainer(BaseSentencePairTrainer): pass


class SentenceTrainer(SentencePairTrainer): pass


class BaseModel(nn.Module):

    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 initial_embeddings, num_classes, mlp_dim,
                 embedding_keep_rate, classifier_keep_rate,
                 use_sentence_pair=False,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.model_dim = model_dim

        args = Args()
        args.size = model_dim
        args.input_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(args.size, vocab.size,
                        vectors=vocab.vectors, make_buffers=False,
                        embedding_dropout_rate=args.input_dropout_rate,
                        )

        self.rnn = nn.LSTM(word_embedding_dim, model_dim, num_layers=1, batch_first=True)

        mlp_input_dim = word_embedding_dim * 2 if use_sentence_pair else model_dim

        self.l0 = nn.Linear(mlp_input_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)

    def run_rnn(self, x, train):
        batch_size, seq_len, model_dim = x.data.size()

        num_layers = 1
        bidirectional = False
        bi = 2 if bidirectional else 1
        h0 = Variable(to_gpu(torch.zeros(num_layers * bi, batch_size, self.model_dim)), volatile=not train)
        c0 = Variable(to_gpu(torch.zeros(num_layers * bi, batch_size, self.model_dim)), volatile=not train)

        # Expects (input, h_0):
        #   input => batch_size x seq_len x model_dim
        #   h_0   => (num_layers x num_directions[1,2]) x batch_size x model_dim
        #   c_0   => (num_layers x num_directions[1,2]) x batch_size x model_dim
        output, (hn, cn) = self.rnn(x, (h0, c0))

        return hn

    def run_mlp(self, h, train):
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = h
        return y


class SentencePairModel(BaseModel):

    def build_example(self, sentences, transitions, train):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(x), volatile=not train))

    def forward(self, sentences, transitions, y_batch=None, train=True, **kwargs):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = self.build_example(sentences, transitions, train)

        emb = self.embed(x)

        hh = torch.squeeze(self.run_rnn(emb, train))
        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        output = self.run_mlp(h, train)

        return output


class SentenceModel(BaseModel):

    def build_example(self, sentences, transitions, train):
        return to_gpu(Variable(torch.from_numpy(x), volatile=not train))

    def forward(self, sentences, transitions, y_batch=None, train=True, **kwargs):
        # Build Tokens
        x = self.build_example(sentences, transitions, train)

        emb = self.embed(x)

        h = torch.squeeze(self.run_rnn(emb, train))
        output = self.run_mlp(h, train)

        return output
