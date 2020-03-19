"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils
from model import layers
from model.nas_rnn import DARTSModel
from model.blocks import *
from model.cpg_modules import ContextualParameterGenerator
from model.link_prediction_models import *


def choose_fact_checker(params):
    name = params['name'].lower()
    if name == 'distmult':
        fact_checker = DistMult(params)
    elif name == 'conve':
        fact_checker = ConvE(params)
    elif name == 'complex':
        fact_checker = Complex(params)
    else:
        raise ValueError('Only, {distmult, conve, and complex}  are supported')
    return fact_checker

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = PositionAwareRNN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

        self.reg_params = opt['reg_params']
        if self.reg_params is not None and self.reg_params['type'] == 'fact_checking':

            self.fact_checker = choose_fact_checker(self.reg_params)

    def maybe_place_batch_on_cuda(self, batch):
        base_batch = batch['base'][:7]
        labels = batch['base'][7]
        orig_idx = batch['base'][8]
        if self.opt['cuda']:
            base_batch = [component.cuda() for component in base_batch]
            labels = labels.cuda()
            for name, data in batch['supplemental'].items():
                batch['supplemental'][name] = [component.cuda() for component in data]

        batch['base'] = base_batch
        return batch, labels, orig_idx

    def apply_fact_checking_regularization(self, inputs, sentence_encs, token_encs):
        subj_masks, obj_masks = inputs['supplemental']['entity_masks']
        masks = inputs['base'][1]
        batch_size, num_tokens = subj_masks.shape
        # remove mask out subjects and objects in sentence masks
        non_entity_masks = masks.eq(constant.PAD_ID)
        non_entity_masks.masked_fill_(subj_masks.bool(), constant.PAD_ID)
        non_entity_masks.masked_fill_(obj_masks.bool(), constant.PAD_ID)
        # extract subject and object representations
        # [B, T, E], [B, T, E] --> [B, S, E], [B, O, E] --> [B, E], [B, E]
        # subj_outputs = token_encs.masked_fill((1 - subj_masks).view(batch_size, -1, 1).bool(), float('-inf'))
        subj_outputs = (token_encs + (subj_masks.view(batch_size, -1, 1) + 1e-45).log())
        # outputs * subj_masks.unsqueeze(-1)
        # obj_outputs = token_encs.masked_fill((1 - obj_masks).view(batch_size, -1, 1).bool(), float('-inf'))
        obj_outputs = (token_encs + (obj_masks.view(batch_size, -1, 1) + 1e-45).log())
        # obj_outputs = outputs * obj_masks.unsqueeze(-1)
        subj_outputs = subj_outputs.max(1, keepdim=True)[0]
        obj_outputs = obj_outputs.max(1, keepdim=True)[0]
        # reshape sentence encoding for compatibility with fact checker
        sentence_encs = sentence_encs.view(batch_size, 1, -1)
        closeness = self.fact_checker(subj_outputs, sentence_encs, obj_outputs)
        closeness = closeness.reshape(-1)
        closeness = F.logsigmoid(closeness)
        return - closeness


    def update(self, batch):
        """ Run a step of forward and backward model update. """
        inputs, labels, _ = self.maybe_place_batch_on_cuda(batch)
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, sentence_encs, token_encs = self.model(inputs)
        loss = self.criterion(logits, labels)

        if self.reg_params is not None and self.reg_params['type'] == 'fact_checking':
            regularization_measure = self.apply_fact_checking_regularization(inputs=inputs,
                                                                             sentence_encs=sentence_encs,
                                                                             token_encs=token_encs)
            loss += self.reg_params['lambda'] * regularization_measure.sum()


        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs, labels, orig_idx = self.maybe_place_batch_on_cuda(batch)
        # forward
        self.model.eval()
        logits, _, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

class PositionAwareRNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()

        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        # Using BiLSTM or LSTM
        if opt['encoding_type'].lower() in ['bilstm', 'lstm']:
            self.bidirectional_encoding = opt['bidirectional_encoding']

        self.encoding_dim = opt['encoding_dim']

        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])
        
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,
            dropout=opt['dropout'], bidirectional=self.bidirectional_encoding)

        if opt['attn']:
            self.attn_layer = layers.PositionAwareAttention(self.encoding_dim,
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])

        elif opt['fact_checking_attn']:
            self.fact_checker = choose_fact_checker(opt['fact_checker_params'])
            # Condense outputs to fit fact checker dimensions
            embedding_dim = self.fact_checker.emb_dim1 * self.fact_checker.emb_dim2
            if self.fact_checker.is_pretrained and embedding_dim != opt['encoding_dim']:
                self.token_encoder = nn.Linear(opt['encoding_dim'], embedding_dim)
                self.subj_encoder = nn.Linear(opt['encoding_dim'], embedding_dim)
                self.obj_encoder = nn.Linear(opt['encoding_dim'], embedding_dim)
                self.encode_fact_check_inputs = True
            else:
                self.encode_fact_check_inputs = False

        self.linear = nn.Linear(self.encoding_dim, opt['num_class'])

        self.opt = opt
        self.topn = float(self.opt.get('topn', 1e10))
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data[:-2].copy_(self.emb_matrix)

        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
        if self.opt['fact_checking_attn']:
            self.sent_linear.weight.data.normal_(std=0.001)
            self.output_linear.weight.data.normal_(std=0.001)
            self.position_linear.weight.data.normal_(std=0.001)

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size):
        num_layers = self.opt['num_layers']
        if self.bidirectional_encoding:
            num_layers *= 2
        state_shape = (num_layers,
                       batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs):
        base_inputs, supplemental_inputs = inputs['base'], inputs['supplemental']
        words, masks, pos, ner, deprel, subj_pos, obj_pos = base_inputs
        if self.opt['fact_checking_attn']:
            subj_masks, obj_masks = inputs['supplemental']['entity_masks']
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]
        
        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]
        inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input

        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
        outputs = self.drop(outputs)

        # attention
        if self.opt['attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)


            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)

        elif self.opt['fact_checking_attn']:
            # remove mask out subjects and objects in sentence masks
            non_entity_masks = masks.eq(constant.PAD_ID)
            non_entity_masks.masked_fill_(subj_masks.bool(), constant.PAD_ID)
            non_entity_masks.masked_fill_(obj_masks.bool(), constant.PAD_ID)
            # extract subject and object representations
            # [B, T, E], [B, T, E] --> [B, S, E], [B, O, E] --> [B, E], [B, E]
            subj_outputs = outputs.masked_fill((1 - subj_masks).view(batch_size, -1, 1).bool(), float('-inf'))
                # outputs * subj_masks.unsqueeze(-1)
            obj_outputs = outputs.masked_fill((1 - obj_masks).view(batch_size, -1, 1).bool(), float('-inf'))
            # obj_outputs = outputs * obj_masks.unsqueeze(-1)
            subj_outputs = subj_outputs.max(1, keepdim=True)[0]
            obj_outputs = obj_outputs.max(1, keepdim=True)[0]
            # Encode inputs to fact checker if needed
            if self.encode_fact_check_inputs:
                outputs = self.token_encoder(outputs)
                subj_outputs = self.subj_encoder(subj_outputs)
                obj_outputs = self.obj_encoder(obj_outputs)

            representation_relevances = self.fact_checker(subj_outputs, outputs, obj_outputs)
            # remove subject and object representations
            masked_elements = non_entity_masks.view(batch_size, -1, 1)
            representation_relevances = representation_relevances + (masked_elements + 1e-45).log()
            indicator_weights = F.softmax(representation_relevances, dim=1)

            final_hidden = (indicator_weights * outputs).sum(dim=1)
        else:
            final_hidden = hidden
        logits = self.linear(final_hidden)
        return logits, final_hidden, outputs
