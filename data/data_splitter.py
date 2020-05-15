import numpy as np
import os
import json
from collections import defaultdict


def load_data(data_file):
    with open(data_file, 'rb') as handle:
        data = json.load(handle)
    return data

def find_blocks(relations):
    blocks = []
    idx = 0
    start_relation = None
    relation_counter = 0
    while idx < len(relations):
        current_relation = relations[idx]
        if start_relation is None:
            start_relation = current_relation
        elif current_relation != start_relation:
            blocks.append((start_relation, relation_counter))
            start_relation = current_relation
            relation_counter = 0
        relation_counter += 1
        idx += 1
    blocks.append((start_relation, relation_counter))
    return blocks

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

def compute_id2info(data):
    id2info = {}
    for d in data:
        d_id = d['id']
        id2info[d_id] = d
    return id2info

def are_splits_correct(orig_data, split1, split2):
    orig_id2info = compute_id2info(orig_data)
    split1_id2info = compute_id2info(split1)
    split2_id2info = compute_id2info(split2)

    incorrect_data = []
    for sample_id in orig_id2info.keys():
        sample_orig = orig_id2info[sample_id]
        if sample_id in split1_id2info:
            sample_split1 = split1_id2info[sample_id]
            if sample_orig != sample_split1:
                incorrect_data.append({'orig': sample_orig, 'split1': sample_split1})
        if sample_id in split2_id2info:
            sample_split2 = split2_id2info[sample_id]
            if sample_orig != sample_split2:
                incorrect_data.append({'orig': sample_orig, 'split2': sample_split2})
        elif sample_id not in split1_id2info and sample_id not in split2_id2info:
            sample_split1 = split1_id2info[sample_id]
            sample_split2 = split2_id2info[sample_id]
            incorrect_data.append({'orig': sample_orig, 'split1': sample_split1, 'split2': sample_split2})
    return incorrect_data


# source_dir = '/usr0/home/gis/data/tacred/data/json/'
source_dir = '/Volumes/External HDD/dataset/tacred/data/json'
file_to_split = os.path.join(source_dir, 'train_negatives.json')
split_proportion = .0

data = load_data(file_to_split)
grouped_data = group_by_triple(data)
small_split, large_split = stratify_sampling(grouped_data, split_prop=split_proportion)
assert(len(are_splits_correct(orig_data=data, split1=large_split, split2=small_split)) == 0)
save_data(small_split, os.path.join(source_dir, f'test_split-{split_proportion}.json'))
save_data(large_split, os.path.join(source_dir, f'train_split-{split_proportion}.json'))

