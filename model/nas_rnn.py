import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from utils import constant, torch_utils
from model import layers
from model.blocks import *

INITRANGE = 0.04

class DARTSModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(DARTSModel, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.emb_dim = opt['emb_dim']
        self.hidden_dim = opt['hidden_dim']
        self.is_training = True

        self.dropout_x = opt['dropout_x']
        self.dropout_h = opt['dropout_h']

        self.token_embs = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        if emb_matrix is not None:
            self.token_embs.weight.data.copy_(self.emb_matrix)

        input_dim = opt['emb_dim']
        if opt['pos_dim'] > 0:
            self.pos_embs = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'], padding_idx=constant.PAD_ID)
            input_dim += opt['pos_dim']
        if opt['ner_dim'] > 0:
            self.pos_embs = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'], padding_idx=constant.PAD_ID)
            input_dim += opt['ner_dim']

        self.input_encoder = nn.Linear(input_dim, opt['emb_dim'])

        self.merge_layers = opt['arc_merge_layers']
        self.connections = opt['arc_connections']
        steps = len(self.connections)
        w0_input_dim = self.emb_dim + self.hidden_dim
        self._W0 = nn.Parameter(torch.Tensor(w0_input_dim, 2 * self.hidden_dim).uniform_(-INITRANGE, INITRANGE))
        self._Ws = nn.ParameterList([
          nn.Parameter(torch.Tensor(self.hidden_dim, 2 * self.hidden_dim).uniform_(-INITRANGE, INITRANGE)) for i in range(steps)
        ])

        self.decoder = nn.Linear(self.hidden_dim, opt['num_class'])

    def mask2d(self, B, D, keep_prob, cuda=True):
        m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
        m = Variable(m, requires_grad=False)
        if cuda:
            m = m.cuda()
        return m

    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs  # unpack
        batch_size, token_dim = words.shape
        masks = masks.unsqueeze(2)
        token_emb = self.token_embs(words)
        network_inputs = [token_emb]
        if self.opt['pos_dim'] > 0:
            pos_emb = self.pos_embs(pos)
            network_inputs.append(pos_emb)
        if self.opt['ner_dim'] > 0:
            ner_emb = self.ner_embs(ner)
            network_inputs.append(ner_emb)
        network_inputs = torch.cat(network_inputs, dim=-1)
        encoded_inputs = self.input_encoder(network_inputs)

        init_hidden, _ = self.zero_state(batch_size)
        encoded_outputs = self.encode_sequence(encoded_inputs, init_hidden)
        masked_outputs = encoded_outputs * (1. - masks)
        aggregated_output = torch.mean(masked_outputs, dim=1)
        logits = self.decoder(aggregated_output)

        return logits, aggregated_output

    def zero_state(self, batch_size):
        state_shape = (batch_size, self.opt['hidden_dim'])
        c0 = torch.zeros(*state_shape, requires_grad=False)
        h0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def encode_sequence(self, inputs, hidden):
        batch_size, sequence_len, input_dim = inputs.shape
        encoded_steps = []
        for step in range(sequence_len):
            input_step = inputs[:, step, :]
            hidden = self.rnn_pass(input_step, hidden)
            encoded_steps.append(hidden)
        encoded_steps = torch.stack(encoded_steps)
        return encoded_steps, encoded_steps[-1].unsqueeze(0)

    def _get_activation(self, name):
      if name == 'tanh':
        f = F.tanh
      elif name == 'relu':
        f = F.relu
      elif name == 'sigmoid':
        f = F.sigmoid
      elif name == 'identity':
        f = lambda x: x
      else:
        raise NotImplementedError
      return f

    def rnn_pass(self, input_step, hidden):
        batch_size, input_dim = input_step.shape
        if self.is_training:
            x_mask = self.mask2d(batch_size, input_dim, keep_prob=1. - self.dropout_x)
            h_mask = self.mask2d(batch_size, input_dim, keep_prob=1. - self.dropout_h)
        else:
            x_mask = h_mask = None
        initial_state = self.compute_initial_state(input_step, hidden, x_mask, h_mask)
        past_states = [initial_state]

        for layer_idx, (activation_name, input_connection) in enumerate(self.connections):
            input_state = past_states[input_connection]
            if self.is_training:
                joint_state = (input_state * h_mask).mm(self._Ws[layer_idx])
            else:
                joint_state = input_state.mm(self._Ws[layer_idx])
            cell_state, hidden_state = torch.split(joint_state, self.hidden_dim)
            cell_state = cell_state.sigmoid()
            activation = self._get_activation(activation_name)
            hidden_state = activation(hidden_state)
            output_state = input_state + cell_state * (hidden_state - input_state)
            past_states += [output_state]

        rnn_output = torch.mean(torch.stack([past_states[i] for i in self.merge_layers], -1), -1)
        return rnn_output

    def compute_initial_state(self, init_input, hidden, x_mask, h_mask):
        if self.is_training:
            input_state = torch.cat([init_input * x_mask, hidden * h_mask], dim=-1)
        else:
            input_state = torch.cat([init_input, hidden], dim=-1)
        cell_state, hidden_state = torch.split(input_state.mm(self._W0), self.hidden_dim, dim=-1)
        cell_state = cell_state.sigmoid()
        hidden_state = hidden_state.tanh()
        hidden_state = hidden + cell_state * (hidden_state - hidden)
        return hidden_state

