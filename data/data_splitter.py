import numpy as np
import os
import json
from collections import defaultdict


def load_data(data_file):
    with open(data_file, 'rb') as handle:
        data = json.load(handle)
    return data

def group_by_triple(data):
    triple2data = defaultdict(lambda: list())
    for example in data:
        relation = example['relation']
        subject_type = example['subj_type']
        object_type = example['obj_type']
        triple = (subject_type, relation, object_type)
        triple2data[triple].append(example)

    for triple, examples in triple2data.items():
        triple2data[triple] = np.array(examples)
    return triple2data

def stratify_sampling(relation2data, split_prop=.3):
    large_split = []
    small_split = []
    for triple, examples in relation2data.items():
        sample_amount = int(len(examples) * split_prop)
        if sample_amount == 0:
            large_split += examples.tolist()
            continue
        example_indices = np.arange(len(examples))
        sampled_indices = np.random.choice(example_indices, sample_amount, replace=False)
        remaining_indices = list(set(example_indices) - set(sampled_indices))
        sampled_examples = examples[sampled_indices]
        remaining_examples = examples[remaining_indices]
        small_split += sampled_examples.tolist()
        large_split += remaining_examples.tolist()
    return small_split, large_split

def save_data(data, save_file):
    with open(save_file, 'w') as handle:
        json.dump(data, handle)

# source_dir = '/usr0/home/gis/data/tacred/data/json/'
source_dir = '/Volumes/External HDD/dataset/tacred/data/json'
file_to_split = os.path.join(source_dir, 'train_negatives.json')
split_proportion = .1

data = load_data(file_to_split)
grouped_data = group_by_triple(data)
small_split, large_split = stratify_sampling(grouped_data, split_prop=split_proportion)
save_data(small_split, os.path.join(source_dir, f'test_split-{split_proportion}.json'))
save_data(large_split, os.path.join(source_dir, f'train_split-{split_proportion}.json'))

