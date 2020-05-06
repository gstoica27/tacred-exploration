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
from model.blocks import *

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = PositionAwareRNN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        main_model_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.parameters = main_model_parameters
        # self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def specify_SCE_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        self.label_fn = lambda x: torch.argmax(x, dim=-1).type(torch.int64)
        if self.opt['cuda']:
            self.criterion.cuda()

    def specify_BCE_criterion(self):
        self.criterion = nn.BCEWithLogitsLoss()
        self.label_fn = lambda x: x
        if self.opt['cuda']:
            self.criterion.cuda()

    def reset_optimizer(self):
        self.optimizer = torch_utils.get_optimizer(self.opt['optim'], self.parameters, self.opt['lr'])

    def reset_decoder(self, num_classes, upstage_mappings=None):
        self.model.reset_decoder(num_classes, upstage_mappings=upstage_mappings)

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

    def update(self, batch):
        losses = {}
        """ Run a step of forward and backward model update. """
        inputs, labels, _ = self.maybe_place_batch_on_cuda(batch)
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, sentence_encs, token_encs, supplemental_losses = self.model(inputs)
        main_loss = self.criterion(logits, labels)

        cumulative_loss = main_loss
        losses['main'] = main_loss.data.item()
        # backward
        cumulative_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        # if self.opt['kg_loss'] is not None:
        #     torch.nn.utils.clip_grad_norm_(self.fact_checker.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        cumulative_loss = cumulative_loss.data.item()
        losses['cumulative'] = cumulative_loss
        return losses

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs, labels, orig_idx = self.maybe_place_batch_on_cuda(batch)
        # forward
        self.model.eval()
        logits, _, _, _ = self.model(inputs)
        # loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy()
        logits = logits.data.cpu().numpy()

        if self.opt['relation_masking']:
            relation_masks = inputs['supplemental']['relation_masks'][0].data.cpu().numpy()
            logits[relation_masks == 0] = -np.inf

        probs = probs.tolist()

        labels = labels.data.cpu().numpy().tolist()
        if unsort:
            _, probs, labels = [list(t) for t in zip(*sorted(zip(orig_idx, probs, labels)))]
        return probs, labels, #loss.data.item()

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
        # Initialize relation embeddings. Note: these will be used as the decoder in the PA-LSTM,
        # Using BiLSTM or LSTM
        if opt.get('encoding_type', 'lstm').lower() in ['bilstm', 'lstm']:
            self.bidirectional_encoding = opt.get('bidirectional_encoding', False)
        # TODO: Remove this soft check
        self.encoding_dim = opt.get('encoding_dim', opt['hidden_dim'])

        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,
            dropout=opt['dropout'], bidirectional=self.bidirectional_encoding)

        if opt['attn']:
            self.attn_layer = layers.PositionAwareAttention(self.encoding_dim,
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

        num_output = opt['num_class']
        self.linear = nn.Linear(self.encoding_dim, num_output)
        #self.register_parameter('class_bias', torch.nn.Parameter(torch.zeros((opt['num_class']))))

        self.opt = opt
        self.topn = float(self.opt.get('topn', 1e10))
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    def reset_decoder(self, num_class, upstage_mappings=None):
        """
        Rest decoder parameters.
        :param num_class: Number of classes in output layer
        :type num_class: int
        :param upstage_mappings: Mappings from weight/bias indices of current decoder to indices of newly initialized
            decoder. The idea here is to translate learned class embeddings to newly initialized ones so information
            is not completely lossed.
        :type upstage_mappings: {previous_class_1: [curr_class_1, curr_class_2, ..., curr_class_n], ...}
        :return: None
        """
        upstage_weight = self.linear.weight.data.cpu()
        upstage_bias = self.linear.bias.data.cpu()
        self.linear = nn.Linear(self.encoding_dim, num_class)
        # {previous_cluster_1: [curr_cluster_1, curr_cluster_2, ..., curr_cluster_n], ...}
        if upstage_mappings is not None:
            linear_weight = np.zeros((num_class, self.encoding_dim), dtype=np.float32)
            linear_bias = np.zeros(num_class, dtype=np.float32)

            for previous_cluster_idx, current_clusters_idxs_set in upstage_mappings.items():
                current_clusters_idxs = list(current_clusters_idxs_set)
                linear_weight[current_clusters_idxs] = upstage_weight[previous_cluster_idx]
                linear_bias[current_clusters_idxs] = upstage_bias[previous_cluster_idx]

            self.linear.weight.data.copy_(torch.from_numpy(linear_weight))
            self.linear.bias.data.copy_(torch.from_numpy(linear_bias))

        else:
            self.linear.bias.data.fill_(0)
            init.xavier_uniform_(self.linear.weight, gain=1)
        if self.opt['cuda']:
            self.linear.cuda()

    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)

        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
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
        else:
            final_hidden = hidden

        supplemental_losses = {}
        logits = self.linear(final_hidden)
        return logits, final_hidden, outputs, supplemental_losses
