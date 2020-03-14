"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
from copy import deepcopy

from utils import constant, helper, vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, use_cuda=False, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.use_cuda = use_cuda
        self.loading_type = opt['loading_type']

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, vocab, opt)
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        # self.labels = [id2label[d[-1]] for d in data]
        self.labels = [id2label[d['relation']] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]

            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            # Extract subj and obj spans
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            if self.loading_type  == 'granular_cpg':
            # Extract subj data
                subj_tokens = deepcopy(map_to_ids(tokens[ss:se+1], vocab.word2id))
                subj_pos = deepcopy(pos[ss:se+1])
                subj_ner = deepcopy(ner[ss:se+1])
                subj_deprel = deepcopy(deprel[ss:se+1])
                subj_data = (subj_tokens, subj_pos, subj_ner, subj_deprel)

                # Extract obj data
                obj_tokens = deepcopy(map_to_ids(tokens[os:oe + 1], vocab.word2id))
                obj_pos = deepcopy(pos[os:oe + 1])
                obj_ner = deepcopy(ner[os:oe + 1])
                obj_deprel = deepcopy(deprel[os:oe+1])
                # obj_data = {'tokens': obj_tokens, 'pos': obj_pos,
                #             'ner': obj_ner, 'deprel': obj_deprel}
                obj_data = (obj_tokens, obj_pos, obj_ner, obj_deprel)

                # anonymize tokens
                tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
                tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
                tokens = map_to_ids(tokens, vocab.word2id)

            elif self.loading_type == 'typed_cpg':
                subj_type = 'SUBJ-' + d['subj_type']
                obj_type = 'OBJ-'+d['obj_type']
                subj_data = (subj_type,)
                obj_data = (obj_type,)
                # anonymize tokens
                tokens[ss:se + 1] = ['SUBJ'] * (se - ss + 1)
                tokens[os:oe + 1] = ['OBJ'] * (oe - os + 1)
            else:
                raise ValueError('Only granular_cpg, typed_cpg and normal are valid')

            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            relation = constant.LABEL_TO_ID[d['relation']]
            # sentence_data = {'tokens': tokens, 'pos': pos, 'ner': ner, 'deprel': deprel,
            #                 'subj_positions': subj_positions, 'obj_positions': obj_positions}
            sentence_data = (tokens, pos, ner, deprel, subj_positions, obj_positions)
            sample_data = {'sentence': sentence_data, 'subj': subj_data, 'obj': obj_data, 'relation': relation}
            processed += [sample_data]
            # processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        #return 50
        return len(self.data)

    def unzip_data(self, data, component):
        fn = lambda x: x[component]
        mapped = map(fn, data)
        return list(zip(*mapped))

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        # Sort all fields by sentence lengths for easier RNN operations
        sentence_lens = np.array(list(map(lambda x: len(x['sentence'][0]), batch)))
        descending_len_idxs = np.argsort(-sentence_lens)
        orig_idx = np.argsort(descending_len_idxs)
        # orig_idx = np.arange(batch_size)[descending_len_idxs]
        sorted_batch = np.array(batch)[descending_len_idxs]
        # sorted_batch, orig_idx = sort_all(batch, sentence_lens)

        # batch = list(zip(*batch))
        # assert len(batch) == 7
        # extract components
        subj_batch = self.unzip_data(data=sorted_batch, component='subj')
        obj_batch = self.unzip_data(data=sorted_batch, component='obj')
        sentence_batch = self.unzip_data(data=sorted_batch, component='sentence')
        assert len(sentence_batch) == 6
        assert len(subj_batch) == 4
        assert len(obj_batch) == 4
        # sort all fields by lens for easy RNN operations
        # lens = [len(x) for x in sentence[0]]
        # sentence, orig_idx = sort_all(sentence, lens)
        # descending_len_idxs = np.argsort(-np.array(lens))
        # word dropout
        if not self.eval:
            sentence_words = [word_dropout(sent, self.opt['word_dropout']) for sent in sentence_batch[0]]
            subj_words = [word_dropout(sent, self.opt['word_dropout']) for sent in subj_batch[0]]
            obj_words = [word_dropout(sent, self.opt['word_dropout']) for sent in obj_batch[0]]
            # words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            # words = batch[0]
            sentence_words = sentence_batch[0]
            subj_words = subj_batch[0]
            obj_words = obj_batch[0]

        # convert to tensors
        def maybe_use_cuda(tensor):
            if self.use_cuda:
                return tensor.cuda()
            return tensor
        sentence_words = get_long_tensor(sentence_words, batch_size, self.use_cuda)
        sentence_pos = get_long_tensor(sentence_batch[1], batch_size, self.use_cuda)
        sentence_ner = get_long_tensor(sentence_batch[2], batch_size, self.use_cuda)
        sentence_deprel = get_long_tensor(sentence_batch[3], batch_size, self.use_cuda)
        sentence_subj_positions = get_long_tensor(sentence_batch[4], batch_size, self.use_cuda)
        sentence_obj_positions = get_long_tensor(sentence_batch[5], batch_size, self.use_cuda)
        sentence_masks = maybe_use_cuda(torch.eq(sentence_words, 0))
        # sentence_data = {'words': sentence_words, 'pos': sentence_pos, 'ner': sentence_ner,
        #                  'deprel': sentence_deprel, 'subj_positions': sentence_subj_positions,
        #                  'obj_positions': sentence_obj_positions, 'masks': sentence_masks}
        sentence_data = (sentence_words, sentence_pos, sentence_ner,
                         sentence_deprel, sentence_subj_positions,
                         sentence_obj_positions, sentence_masks)

        subj_words = get_long_tensor(subj_words, batch_size, self.use_cuda)
        subj_pos = get_long_tensor(subj_batch[1], batch_size, self.use_cuda)
        subj_ner = get_long_tensor(subj_batch[2], batch_size, self.use_cuda)
        subj_deprel = get_long_tensor(subj_batch[3], batch_size, self.use_cuda)
        subj_masks = maybe_use_cuda(torch.eq(subj_words, 0))
        # subj_data = {'words': subj_words, 'pos': subj_pos,
        #              'ner': subj_ner, 'deprel': subj_deprel}
        subj_data = (subj_words, subj_pos, subj_ner, subj_deprel, subj_masks)

        obj_words = get_long_tensor(obj_words, batch_size, self.use_cuda)
        obj_pos = get_long_tensor(obj_batch[1], batch_size, self.use_cuda)
        obj_ner = get_long_tensor(obj_batch[2], batch_size, self.use_cuda)
        obj_deprel = get_long_tensor(obj_batch[3], batch_size, self.use_cuda)
        obj_masks = maybe_use_cuda(torch.eq(obj_words, 0))
        # obj_data = {'words': obj_words, 'pos': obj_pos, 'ner': obj_ner, 'deprel': obj_deprel}
        obj_data = (obj_words, obj_pos, obj_ner, obj_deprel, obj_masks)

        relations = list(map(lambda x:x['relation'], sorted_batch))
        relations = torch.LongTensor(relations)
        # convert to tensors
        # words = get_long_tensor(words, batch_size)
        # masks = torch.eq(words, 0)
        # pos = get_long_tensor(batch[1], batch_size)
        # ner = get_long_tensor(batch[2], batch_size)
        # deprel = get_long_tensor(batch[3], batch_size)
        # subj_positions = get_long_tensor(batch[4], batch_size)
        # obj_positions = get_long_tensor(batch[5], batch_size)

        # rels = torch.LongTensor(batch[6])

        # return (words, masks, pos, ner, deprel, subj_positions, obj_positions, rels, orig_idx)
        return {'sentence': sentence_data, 'subj': subj_data, 'obj': obj_data,
                'relations': relations, 'orig_idx': orig_idx}

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size, use_cuda=False):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    if use_cuda:
        tokens = tokens.cuda()
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

