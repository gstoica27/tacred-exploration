import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils
from model import layers
from model.lstm_pg import PGLSTM

class ConditionalRE(nn.Module):
    def __init__(self, params, word_matrix=None):
        super(ConditionalRE, self).__init__()
        self.dropout = nn.Dropout(params['dropout'])
        self.word_embs = nn.Embedding(params['vocab_size'], 
                                      params['emb_dim'], 
                                      padding_idx=constant.PAD_ID)
        if params['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), params['pos_dim'],
                                        padding_idx=constant.PAD_ID)
        if params['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), params['ner_dim'],
                                        padding_idx=constant.PAD_ID)

        input_size = params['emb_dim'] + params['pos_dim'] + params['ner_dim']
        self.subj_rnn = nn.LSTM(input_size,
                                params['hidden_dim'],
                                params['num_layers'],
                                batch_first=True,
                                dropout=params['dropout'])
        self.obj_rnn = nn.LSTM(input_size,
                               params['hidden_dim'],
                               params['num_layers'],
                               batch_first=True,
                               dropout=params['dropout'])
        cpg_params = params['cpg']
        network_structure = [params['hidden_dim']*2] + cpg_params['network_structure']
        self.sentence_rnn = PGLSTM(input_size=input_size,
                                   hidden_size=params['hidden_dim'], 
                                   num_layers=params['num_layers'], 
                                   context_info={
                                       'network_structure': network_structure,
                                       'dropout': cpg_params['dropout'],
                                       'use_batch_norm': cpg_params['use_batch_norm'],
                                       'batch_norm_momentum': cpg_params['batch_norm_momentum'],
                                       'use_bias': cpg_params['use_bias']
                                   })
        self.decoder = nn.Sequential(
            nn.Linear(params['hidden_dim'], 100),
            nn.ReLU(),
            nn.Linear(100, params['num_class']))
        
        self.params = params
        self.topn = self.params.get('topn', 1e10)
        self.use_cuda = params['cuda']
        self.word_matrix = word_matrix
        self.init_weights() 
    
    def init_weights(self):
        if self.word_matrix is None:
            self.word_embs.weight.data[1:, :].uniform_(-1., 1.)
        else:
            self.word_matrix = torch.from_numpy(self.word_matrix)
            self.word_embs.weight.data.copy_(self.word_matrix)
        if self.params['pos_dim'] > 0:
            self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.params['ner_dim'] > 0:
            self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.word_embs.weight.requires_grad = False
        elif self.topn < self.params['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.word_embs.weight.register_hook(lambda x: \
                                              torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        state_shape = (self.params['num_layers'], batch_size, self.params['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def aggregate_inputs(self, words, pos=None, ner=None):
        word_emb = self.word_embs(words)
        inputs = [word_emb]
        if self.params['pos_dim'] > 0:
            pos_emb = self.pos_emb(pos)
            inputs.append(pos_emb)
        if self.params['ner_dim'] > 0:
            ner_emb = self.ner_emb(ner)
            inputs.append(ner_emb)
        inputs = self.dropout(torch.cat(inputs, dim=2))
        return inputs

    def encode_span(self, span_inputs, span_masks, span_type='subj'):
        word_lengths = span_masks.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        h0, c0 = self.zero_state(batch_size=span_inputs.shape[0])
        _, idx_sort = torch.sort(word_lengths,  dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        sorted_span_inputs = span_inputs.index_select(0, idx_sort)
        sorted_word_lengths = word_lengths[idx_sort]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(sorted_span_inputs,
                                                          sorted_word_lengths,
                                                          batch_first=True)
        if span_type == 'subj':
            rnn = self.subj_rnn
        else:
            rnn = self.obj_rnn
        packed_outputs, (ht, ct) = rnn(packed_inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = outputs.index_select(0, idx_unsort)
        ht = ht.index_select(1, idx_unsort)
        ct = ct.index_select(1, idx_unsort)
        hidden = self.dropout(ht[-1,:,:])
        outputs = self.dropout(outputs)
        return outputs, hidden

    def forward(self, inputs):
        sentence_data, subj_data, obj_data = inputs['sentence'], inputs['subj'], inputs['obj']
        # Parse all components
        (sentence_words, sentence_pos, sentence_ner,
         sentence_deprel, sentence_subj_positions,
         sentence_obj_positions, sentence_masks) = sentence_data
        (subj_words, subj_pos, subj_ner, subj_deprel, subj_masks) = subj_data
        (obj_words, obj_pos, obj_ner, obj_deprel, obj_masks) = obj_data
        sentence_lengths = list(sentence_masks.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = sentence_words.size()[0]
        # Aggregate input types
        sentence_inputs = self.aggregate_inputs(sentence_words, sentence_pos, sentence_ner)
        subj_inputs = self.aggregate_inputs(subj_words, subj_pos, subj_ner)
        obj_inputs = self.aggregate_inputs(obj_words, obj_pos, obj_ner)
        # Encode Subj and Obj
        subj_enc, subj_hidden = self.encode_span(span_inputs=subj_inputs,
                                                 span_masks=subj_masks,
                                                 span_type='subj')
        obj_enc, obj_hidden = self.encode_span(span_inputs=obj_inputs,
                                               span_masks=obj_masks,
                                               span_type='obj')
        packed_sentence = nn.utils.rnn.pack_padded_sequence(sentence_inputs,
                                                            sentence_lengths,
                                                            batch_first=True)
        batch_sizes = packed_sentence.batch_sizes

        context = torch.cat((subj_hidden, obj_hidden), dim=1)
        h0, c0 = self.zero_state(batch_size=batch_size)
        output, (h1, c1) = self.sentence_rnn(input=sentence_inputs,
                                             past_states=(h0, c0),
                                             batch_sizes=batch_sizes,
                                             context=context)
        hidden = self.dropout(h1[-1, :, :])
        logits = self.decoder(hidden)
        return logits, hidden







