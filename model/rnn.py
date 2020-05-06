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
        main_model_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.parameters = main_model_parameters
        # self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        if self.opt['kg_loss'] is not None:
            self.lambda_term = self.opt['kg_loss']['lambda']
            self.lambda_decay = self.opt['kg_loss'].get('lambda_decay', 1.0)

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update_lambda_term(self):
        self.lambda_term *= self.lambda_decay

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

    def kg_criterion(self, batch_inputs, relations):
        supplemental_inputs = batch_inputs['supplemental']
        subjects, labels = supplemental_inputs['knowledge_graph']
        label_smoothing = self.opt['kg_loss']['label_smoothing']
        labels =  ((1.0 - label_smoothing) * labels) + (1.0 / labels.size(1))
        predicted_objects = self.fact_checker.forward(subjects, relations)
        loss = self.fact_checker.loss(predicted_objects, labels)
        return loss

    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes, dtype=torch.float32)
        return y[labels]

    def update(self, batch):
        losses = {}
        """ Run a step of forward and backward model update. """
        inputs, labels, _ = self.maybe_place_batch_on_cuda(batch)
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, sentence_encs, token_encs, supplemental_losses = self.model(inputs)
        # one_hot_labels = self.one_hot_embedding(labels, num_classes=self.opt['num_class'])
        main_loss = self.criterion(logits, labels)
        cumulative_loss = main_loss
        losses['main'] = main_loss.data.item()
        if self.opt['kg_loss'] is not None:
            relation_kg_loss = supplemental_losses['relation']
            sentence_kg_loss = supplemental_losses['sentence']
            cumulative_loss += (relation_kg_loss + sentence_kg_loss) * self.lambda_term
            # losses.update(supplemental_losses)
            relation_kg_loss_value = relation_kg_loss.data.item()
            sentence_kg_loss_value = sentence_kg_loss.data.item()
            losses.update({'relation_kg': relation_kg_loss_value, 'sentence_kg': sentence_kg_loss_value})
            # kg_loss = self.kg_criterion(batch_inputs=inputs, relations=sentence_encs)
            # cumulative_loss += self.opt['kg_loss']['lambda'] * kg_loss.sum()
            # losses['kg'] = kg_loss.data.item()
        if self.opt['entropy_reg'] is not None:
            log_probs = torch.log_softmax(logits, dim=-1)
            neg_logits = log_probs[:, 0]
            pos_logits = torch.logsumexp(log_probs[:, 1:], dim=-1)
            neg_entropy = torch.mean(neg_logits * torch.exp(neg_logits) + pos_logits * torch.exp(pos_logits))
            cumulative_loss += self.opt['entropy_reg'] * neg_entropy
            losses.update({'neg_entropy': neg_entropy})

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
        # one_hot_labels = self.one_hot_embedding(labels, num_classes=self.opt['num_class'])
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        logits = logits.data.cpu().numpy()

        if self.opt['relation_masking']:
            relation_masks = inputs['supplemental']['relation_masks'][0].data.cpu().numpy()
            logits[relation_masks == 0] = -np.inf
        # If the correct prediction is not no relation, "mask out" no relation
        if self.opt['no_relation_masking']:
            logits[labels.data.cpu().numpy() != 0, 0] = -np.inf

        predictions = np.argmax(logits, axis=1).tolist()
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
        # Initialize relation embeddings. Note: these will be used as the decoder in the PA-LSTM,
        # and as the relation embeddings in the KG model simultaneously.
        if opt['kg_loss'] is not None:
            self.rel_emb = nn.Embedding(opt['kg_loss']['model']['num_relations'],
                                        opt['kg_loss']['model']['embedding_dim'])
            self.object_indexes = torch.from_numpy(np.array(opt['obj_idxs']))
            if opt['cuda']:
                self.object_indexes = self.object_indexes.cuda()
            self.kg_model = choose_fact_checker(opt['kg_loss']['model'])
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

        self.linear = nn.Linear(self.encoding_dim, opt['num_class'], bias=False)
        #self.register_parameter('class_bias', torch.nn.Parameter(torch.zeros((opt['num_class']))))

        self.opt = opt
        self.topn = float(self.opt.get('topn', 1e10))
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    # def load_decoder(self, model_path):
    #     state_dict = torch.load(model_path)
    #     relation_embs = state_dict['emb_rel.weight']
    #     self.linear.weight.data.copy_(relation_embs)
    #     if self.opt['kg_loss']['freeze_embeddings']:
    #         self.linear.weight.requires_grad = False

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

        # self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
        # if self.opt['kg_loss'] is not None:
        #     load_path = self.opt['kg_loss']['model']['load_path']
        #     if load_path is not None:
        #         self.load_decoder(load_path)
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

        if self.opt['kg_loss'] is not None:
            subjects, relations, labels, lookup_idxs = supplemental_inputs['knowledge_graph']
            # object indices to compare against
            e2s = self.emb(self.object_indexes)[lookup_idxs]
            # Obtain embeddings
            subject_embs = self.emb(subjects)
            relation_embs = self.rel_emb(relations)
            # Forward pass through both relation and sentence KGLP
            relation_kg_preds = self.kg_model.forward(subject_embs, relation_embs, e2s, lookup_idxs)
            sentence_kg_preds = self.kg_model.forward(subject_embs, final_hidden, e2s, lookup_idxs)
            # Compute each loss term
            relation_kg_loss = self.kg_model.loss(relation_kg_preds, labels)
            sentence_kg_preds = self.kg_model.loss(sentence_kg_preds, labels)
            supplemental_losses = {'relation':relation_kg_loss, 'sentence': sentence_kg_preds}
            # Remove gradient from flowing to the relation embeddings in the main loss calculation
            logits = torch.mm(final_hidden, self.rel_emb.weight.transpose(1, 0).detach())
            #logits += self.class_bias
        else:
            supplemental_losses = {}
            logits = self.linear(final_hidden)
        return logits, final_hidden, outputs, supplemental_losses
