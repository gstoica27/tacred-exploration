import json
import random
import torch
import numpy as np
import os
from copy import deepcopy
from collections import defaultdict

from utils import constant, helper, vocab


class DataProcessor(object):
    def __init__(self, config, vocab, data_dir, partition_names=['train', 'dev', 'test']):
        self.config = config
        self.data_dir = data_dir
        self.partition_names = partition_names
        self.name2id = {
            'ent2id': vocab.word2id,
            'rel2id': {},
            'binary_rel2id': {'no_relation': 0, 'has_relation': 1},
            'pos2id': constant.POS_TO_ID,
            'ner2id': constant.NER_TO_ID,
            'deprel2id': constant.DEPREL_TO_ID
                        }
        self.graph = {}
        self.preprocess_data()

    def preprocess_data(self):
        partitions = defaultdict(list)
        for partition_name in self.partition_names:
            partition_file = os.path.join(self.data_dir, partition_name + '.json')
            with open(partition_file, 'rb') as handle:
                partition_data = json.load(handle)
                partition_parsed = self.parse_data(partition_data)
                partitions[partition_name] = partition_parsed
        self.partitions = partitions

    def parse_data(self, data):
        parsed_data = []
        num_rel = 0
        for idx, d in enumerate(data):
            tokens = d['token']
            if self.config['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
            tokens = self.map_to_ids(tokens, self.name2id['ent2id'])
            pos = self.map_to_ids(d['stanford_pos'], self.name2id['pos2id'])
            ner = self.map_to_ids(d['stanford_ner'], self.name2id['ner2id'])
            deprel = self.map_to_ids(d['stanford_deprel'], self.name2id['deprel2id'])
            l = len(tokens)
            subj_positions = self.get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = self.get_positions(d['obj_start'], d['obj_end'], l)

            relation_name = d['relation']
            if self.config['typed_relations']:
                relation_name = '{}:{}:{}'.format(d['subj_type'], d['relation'], d['obj_type'])
            if relation_name not in self.name2id['rel2id']:
                # This will fail for typed relations!!
                if 'no_relation' in relation_name:
                   self.name2id['rel2id'][relation_name] = 41
                else:
                    self.name2id['rel2id'][relation_name] = num_rel
                    num_rel += 1

            if 'no_relation' in relation_name:
                binary_relation = 'no_relation'
            else:
                binary_relation = 'has_relation'
            if binary_relation not in self.name2id['binary_rel2id']:
                self.name2id['binary_rel2id'][binary_relation] = len(self.name2id['binary_rel2id'])

            relation_id = self.name2id['rel2id'][relation_name]
            base_sample = {'tokens': tokens,
                           'pos': pos,
                           'ner': ner,
                           'deprel': deprel,
                           'subj_positions': subj_positions,
                           'obj_positions': obj_positions,
                           'relation': relation_id}
            base_sample = base_sample

            supplemental_sample = {}
            if self.config['relation_masking']:
                # Find all possible correct relations, and mask out those which do not appear in training set
                subject_type = 'SUBJ-' + d['subj_type']
                object_type = 'OBJ-' + d['obj_type']
                subject_id = self.name2id['ent2id'][subject_type]
                object_id = self.name2id['ent2id'][object_type]
                # Relation Graph doesn't exist yet. Complete it
                pair = (subject_id, object_id)
                if pair not in self.graph:
                    self.graph[pair] = set()
                self.graph[pair].add(relation_id)

                supplemental_sample['relation_masking'] = (subject_id, relation_id, object_id)

            parsed_sample = {'base': base_sample, 'supplemental': supplemental_sample}

            parsed_data.append(parsed_sample)

        return parsed_data

    def get_positions(self, start_idx, end_idx, length):
        """ Get subj/obj position sequence. """
        return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
               list(range(1, length - end_idx))

    def map_to_ids(self, names, mapper):
        return [mapper[t] if t in mapper else constant.UNK_ID for t in names]


    def create_iterator(self, config, partition_name='train'):
        partition_data = self.partitions[partition_name]
        cleaned_data = []
        is_eval = True if partition_name != 'train' else False
        if self.config['sample_size'] is not None:
            partition_data = self.perform_stratified_sampling(partition_data)
        if config['binary_classification']:
            id2label = dict([(v, k) for k, v in self.name2id['binary_rel2id'].items()])
        else:
            id2label = dict([(v, k) for k, v in self.name2id['rel2id'].items()])

        for raw_sample in partition_data:
            sample = deepcopy(raw_sample)
            if config['binary_classification']:
                sample_relation = sample['base']['relation']
                if  self.name2id['rel2id']['no_relation'] == sample_relation:
                    sample['base']['relation'] = self.name2id['binary_rel2id']['no_relation']
                else:
                    sample['base']['relation'] = self.name2id['binary_rel2id']['has_relation']

            if config['exclude_negative_data'] and 'no_relation' in id2label[sample['base']['relation']]:
                continue
            if config['relation_masking']:
                subject, _, object = sample['supplemental']['relation_masking']
                known_relations = self.graph[(subject, object)]
                sample['supplemental']['relation_masking'] = (known_relations,)
            cleaned_data.append(sample)

        return Batcher(dataset=cleaned_data,
                       config=self.config,
                       id2label=id2label,
                       is_eval=is_eval,
                       batch_size=self.config['batch_size'])

    def perform_stratified_sampling(self, data):
        sample_size = self.config['sample_size']
        class2size = self.distribute_sample_size(sample_size)
        class2indices = self.group_by_class(data)
        class2sample = self.sample_by_class(class2indices, class2size)
        sample_indices = self.aggregate_sample_indices(class2sample)
        sample_data = np.array(data)[sample_indices]
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
            relation = sample['base']['relation']
            if relation not in class2indices:
                class2indices[relation] = []
            class2indices[relation].append(idx)
        return class2indices

    def distribute_sample_size(self, sample_size):
        class2size = {}
        class_sample_size = int(sample_size / len(self.name2id['rel2id']))
        remainder = sample_size % len(self.name2id['rel2id'])
        class_sample_bonus = np.random.choice(len(self.name2id['rel2id']), remainder, replace=False)
        for rel_id in self.name2id['rel2id'].values():
            class2size[rel_id] = class_sample_size
            if rel_id in class_sample_bonus:
                class2size[rel_id] += 1
        return class2size




class Batcher(object):
    def __init__(self, dataset, config, id2label, batch_size=50, is_eval=False):
        self.id2label = id2label
        self.batch_size = batch_size
        self.is_eval = is_eval
        self.config = config

        if not self.is_eval:
            np.random.shuffle(dataset)

        self.labels = [id2label[d['base']['relation']] for d in dataset]
        self.num_examples = len(dataset)
        self.batches = self.create_batches(dataset, batch_size=batch_size)

    def create_batches(self, data, batch_size=50):
        batched_data = []
        for batch_start in range(0, len(data), batch_size):
            batch_end = batch_start + batch_size
            batch = data[batch_start: batch_end]
            # merge base batch components
            base_batch = list(map(lambda sample: self.base_mapper(sample['base']), batch))
            # merge supplemental components
            supplemental_names = data[0]['supplemental'].keys()
            supplemental = dict()
            for name in supplemental_names:
                supplemental[name] = list(map(lambda sample: sample['supplemental'][name], batch))
            
            data_batch = {'base': base_batch, 'supplemental': supplemental}
            batched_data.append(data_batch)
        return batched_data

    def base_mapper(self, sample):
        return (
            sample['tokens'],
            sample['pos'],
            sample['ner'],
            sample['deprel'],
            sample['subj_positions'],
            sample['obj_positions'],
            sample['relation']
        )
    
    def ready_base_batch(self, base_batch, batch_size):
        batch = list(zip(*base_batch))
        assert len(batch) == 7

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = self.sort_all(batch, lens)

        # word dropout
        if not self.is_eval:
            words = [self.word_dropout(sent, self.config['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = self.get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = self.get_long_tensor(batch[1], batch_size)
        ner = self.get_long_tensor(batch[2], batch_size)
        deprel = self.get_long_tensor(batch[3], batch_size)
        subj_positions = self.get_long_tensor(batch[4], batch_size)
        obj_positions = self.get_long_tensor(batch[5], batch_size)

        rels = torch.LongTensor(batch[6])

        merged_components = (words, masks, pos, ner, deprel, subj_positions, obj_positions, rels, orig_idx)
        return {'base': merged_components, 'sentence_lengths': lens}

    def ready_masks_batch(self, masks_batch, batch_size, sentence_lengths):
        batch = list(zip(*masks_batch))
        # sort all fields by lens for easy RNN operations
        batch, _ = self.sort_all(batch, sentence_lengths)

        subj_masks = self.get_long_tensor(batch[0], batch_size)
        obj_masks = self.get_long_tensor(batch[1], batch_size)
        merged_components = (subj_masks, obj_masks)
        return merged_components

    def ready_relation_masks_batch(self, mask_batch, sentence_lengths):
        num_rel = len(self.rel2id)
        batch = list(zip(*mask_batch))
        known_relations, _ = self.sort_all(batch, sentence_lengths)
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
        batch_labels, _ = self.sort_all(batch, sentence_lengths)
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

    def ready_binary_classification_batch(self, label_batch, sentence_lengths):
        batch = list(zip(*label_batch))
        batch_labels, _ = self.sort_all(batch, sentence_lengths)
        labels = torch.LongTensor(batch_labels[0])
        return (labels,)

    def ready_data_batch(self, batch):
        batch_size = len(batch['base'])
        readied_batch = self.ready_base_batch(batch['base'], batch_size)
        readied_batch['supplemental'] = dict()
        readied_supplemental = readied_batch['supplemental']
        for name, supplemental_batch in batch['supplemental'].items():
            if name == 'relation_masks':
                readied_supplemental[name] = self.ready_relation_masks_batch(
                    mask_batch=supplemental_batch,
                    sentence_lengths=readied_batch['sentence_lengths'])
            elif name == 'binary_labels':
                readied_supplemental[name] = self.ready_binary_labels_batch(
                    label_batch=supplemental_batch,
                    sentence_lengths=readied_batch['sentence_lengths'])
        return readied_batch

    def get_long_tensor(self, tokens_list, batch_size):
        """ Convert list of list of tokens to a padded LongTensor. """
        token_len = max(len(x) for x in tokens_list)
        tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
        return tokens

    def word_dropout(self, tokens, dropout):
        """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
        return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
                    else x for x in tokens]

    def sort_all(self, batch, lens):
        """ Sort all fields by descending order of lens, and return the original indices. """
        unsorted_all = [lens] + [range(len(lens))] + list(batch)
        sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
        return sorted_all[2:], sorted_all[1]
    
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError
        if item < 0 or item >= self.__len__():
            raise IndexError
        batch = self.batches[item]
        batch = self.ready_data_batch(batch)
        return batch
        
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)