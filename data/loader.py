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
                 kg_graph=None, rel_graph=None, exclude_triples=set()):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.entities = set()
        self.relations = set()
        # Triples to exclude for Triple isolation checking
        self.exclude_triples = exclude_triples
        # Knowledge graph vocabulary (optional)
        if self.opt['kg_loss'] is not None:
            # Extract file name without path or extension
            self.partition_name = os.path.splitext(os.path.basename(filename))[0]
            if kg_graph is not None:
                self.kg_graph = deepcopy(kg_graph)
                # Add entities and relations to data partition
                for (e1, rel), e2s in self.kg_graph.items():
                    self.entities.add(-e1)
                    self.entities = self.entities.union(e2s)
                    self.relations.add(rel)
            else:
                self.kg_graph = defaultdict(lambda: set())
        else:
            self.kg_graph = None

        # if self.opt['relation_masking']:
        # Graph already created in training dataset. Don't create new one b/c it wouldn't be correct.
        if rel_graph is not None:
            self.e1e2_to_rel = rel_graph
            self.rel_graph_pre_exists = True
        else:
            self.e1e2_to_rel = defaultdict(lambda: set())
            self.rel_graph_pre_exists = False
        # else:
        #     self.e1e2_to_rel = None

        self.eval = evaluation
        self.remove_entity_types = opt['remove_entity_types']

        with open(filename) as infile:
            data = json.load(infile)
        np.random.shuffle(data)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)
        # shuffle for training
        if not evaluation:
            if self.opt['negative_factor'] is not None:
                data = self.downsample_data(data, self.opt['negative_factor'], partition='negatives')
            if self.opt['positive_factor'] is not None:
                data = self.downsample_data(data, self.opt['positive_factor'], partition='positives')
            data = self.shuffle_data(data)

        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data['base']]
        self.num_examples = len(data['base'])
        # chunk into batches
        data = self.create_batches(data=data, batch_size=batch_size)
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def downsample_data(self, data, factor=4.0, partition='negatives'):
        if partition == 'negatives':
            num_desired = min(len(data['base']), self.total_positives * factor)
            num_triples = len([t for t in self.triple2data_idxs if t[1] == 'no_relation'])
        else:
            num_desired = min(len(data['base']), self.total_negatives * factor)
            num_triples = len([t for t in self.triple2data_idxs if t[1] != 'no_relation'])

        per_triple_sample_num = int(num_desired / num_triples)
        sorted_triple2data_idxs = {k: v for k, v in sorted(self.triple2data_idxs.items(),
                                                           key=lambda item: len(item[1]))}
        chosen_idxs = []
        num_roi = 0
        for (triple, data_idxs) in sorted_triple2data_idxs.items():
            _, relation, _ = triple
            if relation != 'no_relation' and partition == 'negatives':
                chosen_idxs += data_idxs
            elif relation == 'no_relation' and partition == 'positives':
                chosen_idxs += data_idxs
            else:
                sample_num = min(len(data_idxs), per_triple_sample_num)
                num_roi += sample_num
                sampled_idxs = np.random.choice(data_idxs, sample_num, replace=False).tolist()
                chosen_idxs += sampled_idxs
                num_triples -= 1
                if num_triples > 0:
                    per_triple_sample_num = int((num_desired - num_roi) / num_triples)
        chosen_base_data = np.array(data['base'])[chosen_idxs]
        chosen_supplemental_data = {}
        for name, other_data in data['supplemental'].items():
            chosen_supplemental_data[name] = np.array(other_data)[chosen_idxs]
        chosen_data = {'base': chosen_base_data, 'supplemental': chosen_supplemental_data}
        return chosen_data

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
        self.total_positives = 0
        self.total_negatives = 0
        self.triple2data_idxs = defaultdict(lambda: list())
        for idx, d in enumerate(data):
            subject_type = d['subj_type']
            object_type = d['obj_type']
            relation = d['relation']

            self.triple2data_idxs[(subject_type, relation, object_type)].append(idx)

            if relation != 'no_relation':
                self.total_positives += 1
            else:
                self.total_negatives += 1
            # Exclude triple occurrences
            if (subject_type, relation, object_type) in self.exclude_triples and not self.eval:
                num_excluded += 1
                self.triple_idxs.append(idx)
                continue
            # Store idxs corresponding to triples excluded in training, in eval.
            elif (subject_type, relation, object_type) in self.exclude_triples and self.eval:
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
            relation = constant.LABEL_TO_ID[d['relation']]

            base_processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation)]
            # Use KG component
            if self.opt['kg_loss'] is not None:
                subject_type = 'SUBJ-' + d['subj_type']
                object_type = 'OBJ-' + d['obj_type']
                if not self.eval:
                    self.entities.add(subject_type)
                    self.entities.add(object_type)
                    self.relations.add(relation)
                # Subtract offsets where needed. The way this works is that to find the corresponding subject or
                # object embedding from the tokens, an embedding lookup is performed on the pretrained word2vec
                # embedding matrix. The lookup only involves the subject, so the corresponding mapping utilizes
                # the original subject token position in the vocab. However, the object ids will yield binary
                # labels indicating whether a respective object is a valid answer to the (subj, rel) pair. Thus,
                # We offset the object id so that it results in a zero-indexed binary labeling downstream. Note,
                # the offset is 4 because the vocab order is: ['PAD', 'UNK', 'SUBJ-_', 'SUBJ-_', 'OBJ-*']. So
                # objects are at index 4 onwards.
                subject_id = vocab.word2id[subject_type]
                object_id = vocab.word2id[object_type] - 4
                self.kg_graph[(subject_id, relation)].add(object_id)
                supplemental_components['knowledge_graph'] += [(subject_id, relation, object_id)]
                # Extract all known answers for subject type, relation pair in KG
                # supplemental_components['knowledge_graph'] += [(subject_id, known_object_types)]
            # if self.opt['relation_masking']:
            # Find all possible correct relations, and mask out those which do not appear in training set
            subject_type = 'SUBJ-' + d['subj_type']
            object_type = 'OBJ-' + d['obj_type']
            subject_id = vocab.word2id[subject_type]
            object_id = vocab.word2id[object_type] - 4
            # Relation Graph doesn't exist yet. Complete it
            if not self.rel_graph_pre_exists:
                self.e1e2_to_rel[(subject_id, object_id)].add(relation)
            supplemental_components['relation_masks'] += [(subject_id, relation, object_id)]

        if self.opt['kg_loss'] is not None:
            component_data = supplemental_components['knowledge_graph']
            for idx in range(len(component_data)):
                instance_subj, instance_rel, instance_obj = component_data[idx]
                known_objects = self.kg_graph[(instance_subj, instance_rel)]
                component_data[idx] = (instance_subj, instance_rel, known_objects)

        # if self.opt['relation_masking']:
        component_data = supplemental_components['relation_masks']
        for idx in range(len(component_data)):
            instance_subj, instance_rel, instance_obj = component_data[idx]
            known_relations = self.e1e2_to_rel[(instance_subj, instance_obj)]
            component_data[idx] = (known_relations,)
        # transform to arrays for easier manipulations
        for name in supplemental_components.keys():
            supplemental_components[name] = np.array(supplemental_components[name])

        seven_rel_indices = []
        for idx, known_rels in enumerate(component_data):
            if len(known_rels[0]) != 7:
                seven_rel_indices.append(idx)
        
        self.seven_rel_indices = seven_rel_indices
        self.raw_data = np.array(self.raw_data)[seven_rel_indices]
        base_processed = np.array(base_processed)[seven_rel_indices]
        supplemental_components['relation_masks'] = np.array(supplemental_components['relation_masks'])[
            seven_rel_indices]

        return {'base': np.array(base_processed), 'supplemental': supplemental_components}

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

    def ready_knowledge_graph_batch(self, kg_batch, sentence_lengths):
        # Offset because we don't include the 2 subject entities
        num_ent = len(self.entities) - 2
        batch = list(zip(*kg_batch))
        batch, _ = sort_all(batch, sentence_lengths)
        subjects, relations, known_objects = batch
        subjects = torch.LongTensor(subjects)
        relations = torch.LongTensor(relations)
        labels = []
        lookup_idxs = []
        for known_objs in known_objects:
            binary_labels = np.zeros(num_ent, dtype=np.float32)
            binary_labels[list(known_objs)] = 1.

            if 'train' in self.partition_name:
                max_labels = self.opt['dataset']['max_labels']
                negative_sample_ratio = self.opt['dataset']['negative_sample_ratio']
                assert  max_labels < num_ent, "max labels: {} must be smaller than num ent: {}".format(
                    max_labels, num_ent
                )
                num_pos = len(known_objs)
                num_neg_needed = negative_sample_ratio * num_pos
                if num_neg_needed + num_pos > max_labels:
                    num_pos_needed = int(num_pos / negative_sample_ratio)
                    num_neg_needed = max_labels - num_pos_needed
                    if num_pos_needed > num_pos:
                        num_neg_needed += num_pos_needed - num_pos
                        num_pos_needed = num_pos
                    if num_neg_needed > num_ent - num_pos:
                        num_neg_needed = num_ent - num_pos
                        num_pos_needed = max_labels - num_neg_needed
                else:
                    num_neg_needed = max_labels - num_pos
                    num_pos_needed = num_pos
                known_objs = list(known_objs)
                np.random.shuffle(known_objs)
                known_idxs = known_objs[:num_pos_needed]
                unknown_objects = np.where(binary_labels == 0)[0]
                np.random.shuffle(unknown_objects)
                unknown_idxs = unknown_objects[:num_neg_needed]
                collective_idxs = np.concatenate((known_idxs, unknown_idxs), axis=0)
                collective_labels = binary_labels[collective_idxs]
                assert len(collective_idxs) == max_labels, 'lookup idxs: {} must match max labels: {}'.format(
                    len(collective_idxs), max_labels
                )
            else:
                collective_labels = binary_labels
                # Make dummy lookup indexes
                collective_idxs = np.ones(num_ent)

            labels.append(collective_labels)
            lookup_idxs.append(collective_idxs)
        labels = np.stack(labels, axis=0)
        labels = torch.FloatTensor(labels)
        lookup_idxs = np.stack(lookup_idxs, axis=0)
        lookup_idxs = torch.LongTensor(lookup_idxs)
        merged_components = (subjects, relations, labels, lookup_idxs)
        return merged_components

    def ready_relation_masks_batch(self, mask_batch, sentence_lengths):
        num_rel = 42
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
            elif name == 'knowledge_graph':
                readied_supplemental[name] = self.ready_knowledge_graph_batch(
                    kg_batch=supplemental_batch,
                    sentence_lengths=readied_batch['sentence_lengths'])
            elif name == 'relation_masks':
                readied_supplemental[name] = self.ready_relation_masks_batch(
                    mask_batch=supplemental_batch,
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

