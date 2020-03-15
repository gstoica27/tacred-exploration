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
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
            subj_type = batch[-2].cuda()
            obj_type = batch[-1].cuda()
            inputs += [subj_type, obj_type]
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]
            subj_type = batch[-2]
            obj_type = batch[-1]
            inputs += [subj_type, obj_type]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
            subj_type = batch[-2].cuda()
            obj_type = batch[-1].cuda()
            inputs += [subj_type, obj_type]
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]
            subj_type = batch[-2]
            obj_type = batch[-1]
            inputs += [subj_type, obj_type]

        orig_idx = batch[8]

        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
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
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
            dropout=opt['dropout'])

        if opt['attn']:
            self.attn_layer = layers.PositionAwareAttention(opt['hidden_dim'],
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

        if opt['use_cpg']:
            num_subj_types = len(constant.SUBJ_NER_TO_ID) - 2
            num_obj_types = len(constant.OBJ_NER_TO_ID) - 2
            self.subj_type_emb = nn.Embedding(num_subj_types, opt['type_dim'])
            self.obj_type_emb = nn.Embedding(num_obj_types, opt['type_dim'])
            self.subj_enc = nn.Linear(opt['type_dim'], opt['type_enc_dim'])
            self.obj_enc = nn.Linear(opt['type_dim'], opt['type_enc_dim'])

            cpg_params = opt['cpg']
            self.cpg_W1 = ContextualParameterGenerator(
                network_structure=[opt['type_enc_dim']*2] + cpg_params['network_structure'],
                output_shape=[opt['hidden_dim'], 100],
                dropout=cpg_params['dropout'],
                use_batch_norm=cpg_params['use_batch_norm'],
                batch_norm_momentum=cpg_params['batch_norm_momentum'],
                use_bias=cpg_params['use_bias'])
            self.cpg_W2 = ContextualParameterGenerator(
                network_structure=[opt['type_enc_dim'] * 2] + cpg_params['network_structure'],
                output_shape=[100, 50],
                dropout=cpg_params['dropout'],
                use_batch_norm=cpg_params['use_batch_norm'],
                batch_norm_momentum=cpg_params['batch_norm_momentum'],
                use_bias=cpg_params['use_bias']
            )
            self.linear = nn.Linear(50, opt['num_class'])
        else:
            self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data[:-2].copy_(self.emb_matrix)

            if self.opt.get('avg_types', False):
                subj_avg_emb = torch.mean(self.emb.weight[self.opt['subj_idxs']], dim=0)
                obj_avg_emb = torch.mean(self.emb.weight[self.opt['obj_idxs']], dim=0)
                self.emb.weight.data[-2].copy_(subj_avg_emb)
                self.emb.weight.data[-1].copy_(obj_avg_emb)

        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

        if self.opt['use_cpg']:
            self.subj_type_emb.weight.data.uniform_(-1., 1.)
            self.obj_type_emb.weight.data.uniform_(-1., 1.)

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
        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
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
        input_size = inputs.size(2)
        
        # rnn
        if self.opt['nas_rnn']:
            hidden = self.zero_state(batch_size)[0][0]
            valid_masks = masks.eq(constant.PAD_ID)
            outputs = self.rnn(inputs, hidden, masks=valid_masks)
            #"""
            # subtract 1 from sequence lens for indexing into output
            seq_idxs = [seq_len - 1 for seq_len in seq_lens]
            # seq_idxs index into the last non-padded element of sequence
            # this indexing will go into the [B, S, E] output tensor -> [B, E] hidden only tensor
            last_valid = outputs[range(batch_size), seq_idxs]
            hidden = self.drop(last_valid)
            outputs = self.drop(outputs)
            #"""
            # hidden = self.drop(outputs)
        else:
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
        else:
            final_hidden = hidden

        if self.opt['use_cpg']:
            if self.opt['difference_type_spaces']:
                # translate for correct zero-indexing
                subj_emb = self.subj_type_emb(subj_type - 2)
                obj_emb = self.obj_type_emb(obj_type - 4)
            else:
                subj_emb = self.emb(subj_type)
                obj_emb = self.emb(obj_type)
            subj_enc = F.relu(self.subj_enc(subj_emb))
            obj_enc = F.relu(self.obj_enc(obj_emb))
            cpg_encs = torch.cat((subj_enc, obj_enc), dim=-1)
            w1 = self.cpg_W1(cpg_encs)
            w2 = self.cpg_W2(cpg_encs)

            hidden1 = F.relu(torch.einsum('ij,ijk->ik', [final_hidden, w1]))
            final_hidden = torch.einsum('ij,ijk->ik', [hidden1, w2])

        logits = self.linear(final_hidden)
        return logits, final_hidden


class ArcRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(ArcRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.prev_weight = nn.Linear(in_features=self.input_size + hidden_size, out_features=2 * hidden_size)
        self.layer1_weight = nn.Linear(in_features=2 * hidden_size, out_features=2 * hidden_size)
        self.layer2_weight = nn.Linear(in_features=2 * hidden_size, out_features=2 * hidden_size)
        self.layer3_weight = nn.Linear(in_features=2 * hidden_size, out_features=2 * hidden_size)
        self.layer4_weight = nn.Linear(in_features=2 * hidden_size, out_features=2 * hidden_size)
        self.layer5_weight = nn.Linear(in_features=2 * hidden_size, out_features=2 * hidden_size)
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None




class ArcModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(ArcModel, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                                        padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                                        padding_idx=constant.PAD_ID)

        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True, dropout=opt['dropout'])
        self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])

        if opt['attn']:
            self.attn_layer = layers.PositionAwareAttention(opt['hidden_dim'],
                                                            opt['hidden_dim'], 2 * opt['pe_dim'], opt['attn_dim'])
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)  # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1)  # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

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
        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs  # unpack
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]

        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]
        inputs = self.drop(torch.cat(inputs, dim=2))  # add dropout to input
        input_size = inputs.size(2)

        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.drop(ht[-1, :, :])  # get the outmost layer h_n
        outputs = self.drop(outputs)

        # attention
        if self.opt['attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        else:
            final_hidden = hidden

        logits = self.linear(final_hidden)
        return logits, final_hidden

