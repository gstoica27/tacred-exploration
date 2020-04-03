import numpy as np
import os
from collections import defaultdict
import json

"""
Summaries to Provide

1. Proportion of overlap between pairwise instances of (subject, relation), (relation, object), (subject, object)
2. Proportion of overlap between triple instances of (subject, relation, object) <-- should be 0%
3. When overlap exists, average overlap between missing element in pair. E.g. overlap between set of relations in 
    training (subject, object) pair and in test of the same pair

"""

def create_partition_graph(data_file):
    data = defaultdict(lambda: defaultdict(lambda: set()))
    with open(data_file, 'r') as handle:
        for line in handle:
            e1, rel, e2 = line.strip().split('\t')
            data['e1rel_e2'][(e1, rel)].add(e2)
            data['e1e2_rel'][(e1, e2)].add(rel)
            data['rele2_e1'][(rel, e2)].add(e1)
            data['e1rele2'][(e1, rel, e2)].add(None)

    return data

def create_full_graph(data_dir):
    file_basenames = ['train', 'valid', 'test']
    graph = defaultdict()
    for basename in file_basenames:
        data_file = os.path.join(data_dir, basename + '.txt')
        graph[basename] = create_partition_graph(data_file)
    return graph

def compute_partition_overlaps(partition_1, partition_2, index):
    partition_1_pairs = set(list(partition_1[index].keys()))
    partition_2_pairs = set(list(partition_2[index].keys()))
    num_overlaps = len(partition_1_pairs.intersection(partition_2_pairs))
    partition_1_coverage = num_overlaps / len(partition_1_pairs)
    partition_2_coverage = num_overlaps / len(partition_2_pairs)
    return {'1': partition_1_coverage, '2': partition_2_coverage}

def compute_partition_link_overlaps(partition_1, partition_2, index):
    partition_1_pairs = set(list(partition_1[index].keys()))
    partition_2_pairs = set(list(partition_2[index].keys()))
    overlaps = partition_1_pairs.intersection(partition_2_pairs)
    partition_1_coverage = []
    partition_2_coverage = []
    for overlap in overlaps:
        answers_1 = partition_1[index][overlap]
        answers_2 = partition_2[index][overlap]
        answers_overlaps = answers_1.intersection(answers_2)
        partition_1_coverage.append(len(answers_overlaps) / len(answers_1))
        partition_2_coverage.append(len(answers_overlaps) / len(answers_2))
    partition_1_coverage = np.mean(partition_1_coverage) if len(partition_1_coverage) else 0.
    partition_2_coverage = np.mean(partition_2_coverage) if len(partition_2_coverage) else 0.
    
    return {'1': partition_1_coverage, '2': partition_2_coverage}

def write_all_statistics(partition_1, partition_2, name_1, name_2):
    e1e2_rel = compute_partition_overlaps(partition_1, partition_2, 'e1e2_rel')
    e1rel_e2 = compute_partition_overlaps(partition_1, partition_2, 'e1rel_e2')
    rele2_e1 = compute_partition_overlaps(partition_1, partition_2, 'rele2_e1')
    e1rele2 = compute_partition_overlaps(partition_1, partition_2, 'e1rele2')
    e1e2_rel_answer = compute_partition_link_overlaps(partition_1, partition_2, 'e1e2_rel')
    e1rel_e2_answer = compute_partition_link_overlaps(partition_1, partition_2, 'e1rel_e2')
    rele2_e1_answer = compute_partition_link_overlaps(partition_1, partition_2, 'rele2_e1')
    print(f'Comparing {name_1} to {name_2}')
    print(f'Overlaps for {name_1} compared to {name_2}')
    print('e1rel to e2: {} | Answers: {}'.format(e1rel_e2['1'], e1rel_e2_answer['1']))
    print('e1e2 to rel: {} | Answers: {}'.format(e1e2_rel['1'], e1e2_rel_answer['1']))
    print('rele2 to e1: {} | Answers: {}'.format(rele2_e1['1'], rele2_e1_answer['1']))
    print('e1rele2: {}'.format(e1rele2['1']))
    print(f'Overlaps for {name_2} compared to {name_1}')
    print('e1rel to e2: {} | Answers: {}'.format(e1rel_e2['2'], e1rel_e2_answer['2']))
    print('e1e2 to rel: {} | Answers: {}'.format(e1e2_rel['2'], e1e2_rel_answer['2']))
    print('rele2 to e1: {} | Answers: {}'.format(rele2_e1['2'], rele2_e1_answer['2']))
    print('e1rele2: {}'.format(e1rele2['2']))

def create_TACRED_kg(data_dir):
    files = ['train', 'dev', 'test']
    graph = defaultdict(lambda: 0)
    triple2data = defaultdict(lambda: list())
    for data_file in files:
        file_path = os.path.join(data_dir, data_file + '.json')
        with open(file_path, 'r') as handle:
            data = json.load(handle)
            for instance in data:
                subject = instance['subj_type']
                object = instance['obj_type']
                relation = instance['relation']
                graph[(subject, relation, object)] += 1
                triple2data[(subject, relation, object)].append(instance)
    return graph, triple2data

def group_by_e1e2(graph):
    e1e2_rel = defaultdict(lambda: set())
    for idx, e1rele2 in enumerate(graph):
        e1, rel, e2 = e1rele2
        e1e2_rel[(e1, e2)].add(rel)
    return e1e2_rel

def group_by_e1rel(e1e2_rel):
    e1rel_e2 = defaultdict(lambda: set())
    for e1e2, rels in e1e2_rel.items():
        e1, e2 = e1e2
        for rel in rels:
            e1rel_e2[(e1, rel)].add(e2)
    return e1rel_e2

def partition_data(nested_dict, max_elems, keep_prob=.1, move_prob=.3):
    partition_1 = defaultdict(lambda: set())
    partition_2 = defaultdict(lambda: set())
    toplevel_shuffled = list(nested_dict.keys())
    np.random.shuffle(toplevel_shuffled)
    for toplevel_key in toplevel_shuffled:
        values = np.array(list(nested_dict[toplevel_key]))
        if max_elems > 0:
            if np.random.random() < keep_prob:
                move_amount = int(len(values) * move_prob)
                if move_amount == 0:
                    move_amount = len(values)
                # Can move at most max elems
                move_amount = min(move_amount, max_elems)
                move_values = set(np.random.choice(values, move_amount, replace=False))
                keep_values = set(values) - move_values
                partition_1[toplevel_key] = partition_1[toplevel_key].union(keep_values)
                partition_2[toplevel_key] = partition_2[toplevel_key].union(move_values)
            else:
                move_amount = len(values)
                move_values = set(np.random.choice(values, move_amount, replace=False))
                partition_2[toplevel_key] = partition_2[toplevel_key].union(move_values)
            max_elems -= move_amount
            max_elems = max(0, max_elems)
        else:
            partition_1[toplevel_key] = partition_1[toplevel_key].union(set(values))

    return partition_1, partition_2

def convert_e1rel_to_e1rele2(e1rel_e2):
    triples = []
    for e1rel, e2s in e1rel_e2.items():
        for e2 in e2s:
            triple = (e1rel[0], e1rel[1], e2)
            triples.append(triple)
    return triples

def convert_e1e2_rel_to_e1rele2(e1e2_rel):
    triples = []
    for e1e2, rels in e1e2_rel.items():
        e1, e2 = e1e2
        for rel in rels:
            triple = (e1, rel, e2)
            triples.append(triple)
    return triples

def ensure_disjoint_e1e2_partitions(e1e2_rel_1, e1e2_rel_2, swap_prob=.5):
    """
    If there is any overlap in (e1, e2) pairs between the training and evaluation datasets,
    move the overlaps from the training to dev sets
    """
    common_e1e2s = set(list(e1e2_rel_1.keys())).intersection(set(list(e1e2_rel_2.keys())))
    for common_e1e2 in common_e1e2s:
        # Transfer all relevant observations in eval dataset to train
        if np.random.random() < swap_prob:
            e1e2_rel_1[common_e1e2] = e1e2_rel_2[common_e1e2].union(e1e2_rel_1[common_e1e2])
            del e1e2_rel_2[common_e1e2]
        # Transfer all relevant observations in train dataset to eval
        else:
            e1e2_rel_2[common_e1e2] = e1e2_rel_2[common_e1e2].union(e1e2_rel_1[common_e1e2])
            del e1e2_rel_1[common_e1e2]


def create_new_TACRED(graph, e1e2_split, e1rel_split, e1rel_e2_split):
    e1e2_rel = group_by_e1e2(graph)
    # There should be no overlap between train and eval partitions for e1e2_rel. Hence keep_prob = 0.0
    max_eval_elems = int(len(graph) * e1e2_split)
    e1e2_rel_train, e1e2_rel_eval = partition_data(e1e2_rel, max_elems=max_eval_elems, keep_prob=-1.)
    # Re-group by e1rel_e2 for next set of partitioning
    e1rel_e2_group = group_by_e1rel(e1e2_rel_train)
    remaining_elems = len(graph) - max_eval_elems
    max_eval_elems = int(remaining_elems * e1rel_split)
    # The partition should be such that the train and eval partitions each have some overlaps,
    #   but have sufficiently disjoint elements when grouped by (e1, rel). This rate at which
    #   we share elements (overlap) or don't (disjoint) is regulated by keep_prob. Additionally,
    #   max_elems specifies the maximum number of data that makes up the eval set.
    e1rel_e2_train, e1rel_e2_eval = partition_data(e1rel_e2_group, max_elems=max_eval_elems, keep_prob=e1rel_e2_split)
    # We now have several partitions, and need to merge them. However, their formats are different
    #    ^need to transform each format to a standard one. Thus, we move all formats back to the graph
    #   format of a list of triples, and then we add the lists together to form each partition.
    # Create Training Set
    train_ds = convert_e1rel_to_e1rele2(e1rel_e2_train)
    # Create Eval Set
    e1e2_rel_eval = convert_e1e2_rel_to_e1rele2(e1e2_rel_eval)
    e1rel_e2_eval = convert_e1rel_to_e1rele2(e1rel_e2_eval)
    eval_ds = e1rel_e2_eval + e1e2_rel_eval
    # Ensure that there is no overlap between train and eval partitions for e1e2_rel.
    eval_e1e2_rel = group_by_e1e2(eval_ds)
    train_e1e2_rel = group_by_e1e2(train_ds)
    ensure_disjoint_e1e2_partitions(train_e1e2_rel, eval_e1e2_rel)
    # Revert partitions to standard format
    train_ds = convert_e1e2_rel_to_e1rele2(train_e1e2_rel)
    eval_ds = convert_e1e2_rel_to_e1rele2(eval_e1e2_rel)
    # Randomly split the eval dataset into Validation and Test partitions.
    # Create Test and Valid Partitions
    eval_len = len(eval_ds)
    eval_idxs = np.arange(eval_len)
    test_idxs = np.random.choice(eval_idxs, int(eval_len * .5), replace=False)
    test_ds = [item for idx, item in enumerate(eval_ds) if idx in test_idxs]
    valid_ds = list(set(eval_ds) - set(test_ds))
    return {'train': train_ds, 'valid': valid_ds, 'test': test_ds}
    # return {'train': train_ds, 'eval': eval_ds}

def transform_partition_graph(partition):
    data = defaultdict(lambda: defaultdict(lambda: set()))
    for (e1, rel, e2) in partition:
        data['e1rel_e2'][(e1, rel)].add(e2)
        data['e1e2_rel'][(e1, e2)].add(rel)
        data['rele2_e1'][(rel, e2)].add(e1)
        data['e1rele2'][(e1, rel, e2)].add(None)

    return data

def compute_partition_size(partition, graph):
    total_size = 0
    for triple in partition:
        triple_freq = graph[triple]
        total_size += triple_freq
    return total_size

def map_graph_to_TACRED_examples(partition, triple2data):
    dataset = []
    for triple in partition:
        triple_data = triple2data[triple]
        dataset += triple_data
    return dataset

def save_dataset(dataset, save_path):
    with open(save_path, 'w') as handle:
        json.dump(dataset, handle)

if __name__ == '__main__':
    # root_dir = '/Users/georgestoica/Desktop/Research/tacred-exploration/temp'
    # dataset_names = ['FB15k-237', 'kinship', 'nell-995', 'nell-995-test', 'umls', 'WN18RR']
    # for dataset_name in dataset_names:
    #     data_dir = os.path.join(root_dir, dataset_name, 'data', dataset_name, dataset_name)
    #     graph = create_full_graph(data_dir)
    #     print('#'*80)
    #     print(f'Dataset: {dataset_name}')
    #     write_all_statistics(partition_1=graph['train'], partition_2=graph['valid'], name_1='train', name_2='valid')
    #     write_all_statistics(partition_1=graph['train'], partition_2=graph['test'], name_1='train', name_2='test')
    #     write_all_statistics(partition_1=graph['valid'], partition_2=graph['test'], name_1='valid', name_2='test')
    tacred_dir = '/Volumes/External HDD/dataset/tacred/data/json'
    e1e2_split = .12
    e1rel_split = .1
    e1rel_e2_split = .5
    tacred_graph, triple2data = create_TACRED_kg(tacred_dir)
    tacred_ic = create_new_TACRED(tacred_graph, e1e2_split, e1rel_split, e1rel_e2_split)

    graph = {'train': transform_partition_graph(tacred_ic['train']),
             'valid': transform_partition_graph(tacred_ic['valid']),
             'test': transform_partition_graph(tacred_ic['test'])
             }
    print('#' * 80)
    print(f'Dataset: TACRED IC')
    write_all_statistics(partition_1=graph['train'], partition_2=graph['valid'], name_1='train', name_2='valid')
    write_all_statistics(partition_1=graph['train'], partition_2=graph['test'], name_1='train', name_2='test')
    write_all_statistics(partition_1=graph['valid'], partition_2=graph['test'], name_1='valid', name_2='test')
    train_size = compute_partition_size(tacred_ic['train'], tacred_graph)
    dev_size = compute_partition_size(tacred_ic['valid'], tacred_graph)
    test_size = compute_partition_size(tacred_ic['test'], tacred_graph)
    print('Train: {} | Dev: {} | Test: {}'.format(train_size, dev_size, test_size))
    train_dataset = map_graph_to_TACRED_examples(tacred_ic['train'], triple2data)
    valid_dataset = map_graph_to_TACRED_examples(tacred_ic['valid'], triple2data)
    test_dataset = map_graph_to_TACRED_examples(tacred_ic['test'], triple2data)
    save_dataset(train_dataset, os.path.join(tacred_dir, 'train_ic.json'))
    save_dataset(test_dataset, os.path.join(tacred_dir, 'test_ic.json'))
    save_dataset(valid_dataset, os.path.join(tacred_dir, 'dev_ic.json'))
