"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab
from collections import defaultdict
import os
from copy import deepcopy

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False,
                 kg_graph=None, rel_graph=None, exclude_triples=set(), rel2id={}):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.entities = set()
        self.relations = set()
        # Triples to exclude for Triple isolation checking
        self.exclude_triples = exclude_triples

        if self.opt['relation_masking']:
            # Graph already created in training dataset. Don't create new one b/c it wouldn't be correct.
            if rel_graph is not None:
                self.e1e2_to_rel = rel_graph
                self.rel_graph_pre_exists = True
            else:
                self.e1e2_to_rel = defaultdict(lambda: set())
                self.rel_graph_pre_exists = False
        else:
            self.e1e2_to_rel = None

        # if self.opt['typed_relations']:
        if self.eval:
            self.rel2id = rel2id
        else:
            self.rel2id = {}

        self.remove_entity_types = opt['remove_entity_types']

        with open(filename) as infile:
            data = json.load(infile)
        np.random.shuffle(data)
        data = self.preprocess(data, vocab, opt)

        if self.opt['sample_size'] is not None:
            data = self.perform_stratified_sampling(data)
        elif self.opt['down_sample'] and not evaluation:
            data = self.distribute_data(data)

        # shuffle for training
        if not evaluation:
            data = self.shuffle_data(data)

        # if self.opt['typed_relations']:
        id2label = dict([(v, k) for k, v in self.rel2id.items()])
        # else:
        #     id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data['base']]
        self.num_examples = len(data['base'])
        # chunk into batches
        data = self.create_batches(data=data, batch_size=batch_size)
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def create_batches(self, data, batch_size):
        batched_data = []
        for batch_start in range(0, len(data['base']), batch_size):

            batch_end = batch_start + batch_size
            base_batch = data['base'][batch_start: batch_end]
            data_batch = {'base': base_batch, 'supplemental': dict()}
            supplemental_batch = data_batch['supplemental']
            for component in data['supplemental']:
                supplemental_batch[component] = data['supplemental'][component][batch_start: batch_end]
            batched_data.append(data_batch)
        return batched_data

    def shuffle_data(self, data):
        indices = list(range(len(data['base'])))
        random.shuffle(indices)
        shuffled_base = data['base'][indices]
        supplemental_data = data['supplemental']
        for name, component in supplemental_data.items():
            supplemental_data[name] = component[indices]
        shuffled_data = {'base': shuffled_base, 'supplemental': supplemental_data}
        return shuffled_data

    def add_entity_mask(self, length, span):
        mask = np.ones(length, dtype=np.float32) * constant.PAD_ID
        mask[span[0]: span[1] + 1] = 1.
        return mask

    def add_entity_masks(self, length, subj_span, obj_span):
        subj_mask = self.add_entity_mask(length=length, span=subj_span)
        obj_mask = self.add_entity_mask(length=length, span=obj_span)
        return (subj_mask, obj_mask)

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        base_processed = []
        supplemental_components = defaultdict(list)
        num_excluded = 0
        self.triple_idxs = []
        filtered_idxs = []
        filtered_data = []
        for idx, d in enumerate(data):
            subject_type = d['subj_type']
            object_type = d['obj_type']
            relation_name = d['relation']
            # Exclude triple occurrences
            if (subject_type, relation_name, object_type) in self.exclude_triples and not self.eval:
                num_excluded += 1
                self.triple_idxs.append(idx)
                continue
            # Store idxs corresponding to triples excluded in training, in eval.
            elif (subject_type, relation_name, object_type) in self.exclude_triples and self.eval:
                self.triple_idxs.append(idx)

            filtered_idxs.append(idx)
            filtered_data.append(d)
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            if self.remove_entity_types:
                tokens[ss:se + 1] = ['SUBJ'] * (se - ss + 1)
                tokens[os:oe + 1] = ['OBJ'] * (oe - os + 1)
            else:
                tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
                tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)

            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)

            if opt['reduce_spans']:
                tokens = reduce_spans(tokens, (ss, se), (os, oe))
                pos = reduce_spans(pos, (ss, se), (os, oe))
                ner = reduce_spans(ner, (ss, se), (os, oe))
                deprel = reduce_spans(deprel, (ss, se), (os, oe))
                l = len(tokens)

                if d['subj_start'] < d['obj_start']:
                    offset = se - ss
                    os -= offset
                    oe = os
                    se = ss
                else:
                    offset = oe - os
                    ss -= offset
                    se = ss
                    oe = os

                subj_positions = get_positions(ss, se, l)
                obj_positions = get_positions(os, oe, l)


            # Create typed "no_relation" relations
            relation_name = d['relation']
            if self.opt['typed_relations']:
                relation_name = '{}:{}:{}'.format(d['subj_type'], d['relation'], d['obj_type'])
            if relation_name not in self.rel2id:
                self.rel2id[relation_name] = len(self.rel2id)
            relation = self.rel2id[relation_name]
            # else:
            #    relation = constant.LABEL_TO_ID[d['relation']]

            base_processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation)]

            if self.opt['relation_masking']:
                # Find all possible correct relations, and mask out those which do not appear in training set
                subject_type = 'SUBJ-' + d['subj_type']
                object_type = 'OBJ-' + d['obj_type']
                subject_id = vocab.word2id[subject_type]
                object_id = vocab.word2id[object_type]
                # Relation Graph doesn't exist yet. Complete it
                if not self.rel_graph_pre_exists:
                    self.e1e2_to_rel[(subject_id, object_id)].add(relation)
                supplemental_components['relation_masks'] += [(subject_id, relation, object_id)]

        if self.opt['relation_masking']:
            component_data = supplemental_components['relation_masks']
            for idx in range(len(component_data)):
                instance_subj, instance_rel, instance_obj = component_data[idx]
                known_relations = self.e1e2_to_rel[(instance_subj, instance_obj)]
                component_data[idx] = (known_relations,)

        # transform to arrays for easier manipulations
        for name in supplemental_components.keys():
            supplemental_components[name] = np.array(supplemental_components[name])

        return {'base': np.array(base_processed), 'supplemental': supplemental_components}

    def perform_stratified_sampling(self, data):
        sample_size = self.opt['sample_size']
        class2size = self.distribute_sample_size(sample_size)
        class2indices = self.group_by_class(data['base'])
        class2sample = self.sample_by_class(class2indices, class2size)
        sample_indices = self.aggregate_sample_indices(class2sample)
        sample_data = {'base': data['base'][sample_indices], 'supplemental': {}}
        for component, component_data in data['supplemental'].items():
            sample_data['supplemental'][component] = component_data[sample_indices]
        return sample_data

    def aggregate_sample_indices(self, class2sample):
        sample_indices = []
        for sample in class2sample.values():
            sample_indices.append(sample)
        sample_indices = np.concatenate(sample_indices)
        return sample_indices

    def sample_by_class(self, class2indices, class2size):
        class2sample = {}
        for relation, indices in class2indices.items():
            sample_size = min(class2size[relation], len(indices))
            sample_indices = np.random.choice(indices, sample_size, replace=False)
            class2sample[relation] = sample_indices
        return class2sample

    def group_by_class(self, data):
        class2indices = {}
        for idx, sample in enumerate(data):
            relation = sample[-1]
            if relation not in class2indices:
                class2indices[relation] = []
            class2indices[relation].append(idx)
        return class2indices

    def distribute_sample_size(self, sample_size):
        class2size = {}
        class_sample_size = int(sample_size / len(self.rel2id))
        remainder = sample_size % len(self.rel2id)
        class_sample_bonus = np.random.choice(len(self.rel2id), remainder, replace=False)
        for rel_id in self.rel2id.values():
            class2size[rel_id] = class_sample_size
            if rel_id in class_sample_bonus:
                class2size[rel_id] += 1
        return class2size

    def distribute_data(self, data):
        pair2rel2data = defaultdict(lambda: defaultdict(list))
        base_data = data['base']
        for idx, d in enumerate(base_data):
            subj_type = np.array(d[0])[np.array(d[4]) == 0][0]
            obj_type = np.array(d[0])[np.array(d[5]) == 0][0]
            relation = d[-1]
            pair2rel2data[(subj_type, obj_type)][relation].append(idx)

        sampled_idxs = []
        for (subj_type, obj_type), rel2data in pair2rel2data.items():
            triple_sizes = list(map(lambda x: len(x), list(rel2data.values())))
            # median_size = self.hard_median(triple_sizes)
            cutoff_size = sorted(triple_sizes)[-2]
            for rel, subset in rel2data.items():
                # Downsample to the be median size
                if len(subset) > cutoff_size:
                    subset_idxs = np.arange(len(subset))
                    selected_idxs = np.random.choice(subset_idxs, cutoff_size, replace=False)
                    new_subset = np.array(subset)[selected_idxs].tolist()
                else:
                    new_subset = subset
                rel2data[rel] = new_subset
                sampled_idxs += new_subset


        sampled_data = {'base': self.filter_data(data['base'], keep_idxs=sampled_idxs), 'supplemental': {}}
        for component in data['supplemental']:
            sampled_data['supplemental'][component] = self.filter_data(data['supplemental'][component],
                                                                       keep_idxs=sampled_idxs)

        return sampled_data

    def filter_data(self, data, keep_idxs):
        keep_idxs = sorted(keep_idxs)
        filtered_data = []
        keep_idx = 0
        for idx, d in enumerate(data):
            if idx == keep_idxs[keep_idx]:
                keep_idx += 1
                filtered_data.append(d)
            if keep_idx >= len(keep_idxs):
                break
        return np.array(filtered_data)

    def hard_median(self, data):
        length = len(data)
        if length % 2 == 0:
            length -= 1
        middle = int(length / 2)
        return sorted(data)[middle]

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def ready_base_batch(self, base_batch, batch_size):
        batch = list(zip(*base_batch))
        assert len(batch) == 7

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        subj_positions = get_long_tensor(batch[4], batch_size)
        obj_positions = get_long_tensor(batch[5], batch_size)

        rels = torch.LongTensor(batch[6])

        merged_components = (words, masks, pos, ner, deprel, subj_positions, obj_positions, rels, orig_idx)
        return {'base': merged_components, 'sentence_lengths': lens}

    def ready_masks_batch(self, masks_batch, batch_size, sentence_lengths):
        batch = list(zip(*masks_batch))
        # sort all fields by lens for easy RNN operations
        batch, _ = sort_all(batch, sentence_lengths)

        subj_masks = get_long_tensor(batch[0], batch_size)
        obj_masks = get_long_tensor(batch[1], batch_size)
        merged_components = (subj_masks, obj_masks)
        return merged_components

    def ready_relation_masks_batch(self, mask_batch, sentence_lengths):
        num_rel = len(self.rel2id)
        batch = list(zip(*mask_batch))
        known_relations, _ = sort_all(batch, sentence_lengths)
        labels = []
        for sample_labels in known_relations[0]:
            binary_labels = np.zeros(num_rel, dtype=np.float32)
            binary_labels[list(sample_labels)] = 1.
            labels.append(binary_labels)
        labels = np.stack(labels, axis=0)
        labels = torch.FloatTensor(labels)
        return (labels,)

    def ready_binary_labels_batch(self, label_batch, sentence_lengths):
        # Remove the no_relation from the mappings. Note, this is only possible because we've
        # guaranteed that no_relation is the last index - so "positive" relations naturally
        # map to the resultant label vector.
        num_rel = len(self.rel2id) - 1
        batch = list(zip(*label_batch))
        batch_labels, _ = sort_all(batch, sentence_lengths)
        labels = []
        for label in batch_labels[0]:
            binary_labels = np.zeros(num_rel, dtype=np.float32)
            # don't add no_relation index because that would be out of bounds.
            if label < num_rel:
                binary_labels[label] = 1.
            labels.append(binary_labels)
        labels = np.stack(labels, axis=0)
        labels = torch.FloatTensor(labels)
        return (labels,)

    def ready_data_batch(self, batch):
        batch_size = len(batch['base'])
        readied_batch = self.ready_base_batch(batch['base'], batch_size)
        readied_batch['supplemental'] = dict()
        readied_supplemental = readied_batch['supplemental']
        for name, supplemental_batch in batch['supplemental'].items():
            if name == 'entity_masks':
                readied_supplemental[name] = self.ready_masks_batch(
                    masks_batch=supplemental_batch,
                    batch_size=batch_size,
                    sentence_lengths=readied_batch['sentence_lengths'])
            elif name == 'relation_masks':
                readied_supplemental[name] = self.ready_relation_masks_batch(
                    mask_batch=supplemental_batch,
                    sentence_lengths=readied_batch['sentence_lengths'])
            elif name == 'binary_labels':
                readied_supplemental[name] = self.ready_binary_labels_batch(
                    label_batch=supplemental_batch,
                    sentence_lengths=readied_batch['sentence_lengths'])
        return readied_batch

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch = self.ready_data_batch(batch)
        return batch


    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def reduce_spans(unabrided_data, subject_span, object_span):
    ss, se = subject_span
    os, oe = object_span
    if ss < os:
        fs, fe = subject_span
        ls, le = object_span
    else:
        fs, fe = object_span
        ls, le = subject_span
    left_data = unabrided_data[:fs]
    middle_data = unabrided_data[fe + 1: ls]
    right_data = unabrided_data[le + 1:]
    return left_data + [unabrided_data[fs]] + middle_data + [unabrided_data[ls]] + right_data


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
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

