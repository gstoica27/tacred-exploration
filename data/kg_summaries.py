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

def partition_data(nested_dict, max_elems, keep_prob=.1, move_prop=.5):
    partition_2 = defaultdict(lambda: set())
    toplevel_shuffled = list(nested_dict.keys())
    # Account for term frequency. This is used to ensure that partition1 retains all INDIVIDUAL
    # information that is in partition2. Note, this doesn't consider pairwise information,
    # which should be (partially) disjoint.
    key0s = defaultdict(lambda: 0)
    key1s = defaultdict(lambda: 0)
    bottom_values = defaultdict(lambda: 0)
    for (key0, key1), value_set in nested_dict.items():
        key0s[key0] += 1
        key1s[key1] += 1
        for value in value_set:
            bottom_values[value] += 1

    np.random.shuffle(toplevel_shuffled)
    while max_elems > 0:
        sample_idx = np.random.choice(len(toplevel_shuffled), 1)[0]
        toplevel_key = toplevel_shuffled[sample_idx]
        key0, key1 = toplevel_key
        values = np.array(list(nested_dict[toplevel_key]))
        # Can only partition values which we have an excess of in partition1.
        available_values = []
        for value in values:
            if bottom_values[value] > 1:
                available_values.append(value)

        move_values = []
        if np.random.random() < keep_prob:

            move_amount = int(len(available_values) * move_prop)
            if move_amount > 0:
                # Can move at most max elems
                move_amount = min(move_amount, max_elems)
                move_values = set(np.random.choice(available_values, move_amount, replace=False))
                # Ensure resultant partition is a subset of the grouped data
                if len(nested_dict[toplevel_key] - partition_2[toplevel_key].union(move_values)) > 0:
                    partition_2[toplevel_key] = partition_2[toplevel_key].union(move_values)
        # Can only move example to second partition as long as ALL top level keys are
        # seen individually in the first partition. This is equivalent to making sure
        # that all entities and relations are observed in the training set in KGs.
        elif not (key0s[key0] == 1 or key1s[key1] == 1 or len(available_values) < len(values)):
            move_amount = len(available_values)
            move_values = set(np.random.choice(available_values, move_amount, replace=False))
            partition_2[toplevel_key] = partition_2[toplevel_key].union(move_values)
            # Main partition as minus one mention of each toplevel component
            key0s[key0] -= 1
            key1s[key1] -= 1
        else:
            move_amount = 0
        max_elems -= move_amount
        # Main partition as minus one mention of each bottom level value
        for value in move_values:
            bottom_values[value] -= 1

    partition_1 = compute_dictionary_difference(nested_dict, partition_2)


    assert len(set(list(map(lambda key: key[1], partition_1.keys())))) == len(key1s), \
            'Partition1 must have 17 objects'
    assert len(set(list(map(lambda key: key[0], partition_1.keys())))) == len(key0s), \
            'Partition1 must have 2 subjects'
    p1_values = set()
    for values in partition_1.values():
        p1_values = p1_values.union(values)
    p2_values = set()
    for values in partition_2.values():
        p2_values = p2_values.union(values)

    assert len(p1_values) == len(bottom_values), 'Partition1 must have 42 relations'
    return partition_1, partition_2

def compute_dictionary_difference(dict1, dict2):
    """This function computes the operation dict1 - dict2"""
    difference_dict = {}
    for key, values in dict1.items():
        if key in dict2:
            value_difference = dict1[key] - dict2[key]
            if len(value_difference) > 0:
                difference_dict[key] = value_difference
        else:
            value_difference = dict1[key]
            difference_dict[key] = value_difference
    print('key1: {} '.format(len(set(list(map(lambda key: key[1], difference_dict))))))
    return difference_dict

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
    e1_counts = defaultdict(lambda: 0)
    e2_counts = defaultdict(lambda: 0)
    relation_counts = defaultdict(lambda: 0)
    for (e1, e2), relations in e1e2_rel_1.items():
        e1_counts[e1] += 1
        e2_counts[e2] += 1
        for relation in relations:
            relation_counts[relation] += 1

    common_e1e2s = set(list(e1e2_rel_1.keys())).intersection(set(list(e1e2_rel_2.keys())))
    for common_e1e2 in common_e1e2s:
        # Ensure that the sample can be removed from the first partition.
        # This can only happen if it does not affect the partition's coverage.
        e1, e2 = common_e1e2
        e1_count = e1_counts[e1]
        e2_count = e2_counts[e2]
        relations = e1e2_rel_1[common_e1e2]
        losable_pair = e1_count > 1 and e2_count > 1
        for relation in relations:
            if relation_counts[relation] == 1:
                losable_pair = False
        # Transfer all relevant observations in eval dataset to train
        if np.random.random() < swap_prob and losable_pair:
            e1e2_rel_2[common_e1e2] = e1e2_rel_2[common_e1e2].union(e1e2_rel_1[common_e1e2])
            del e1e2_rel_1[common_e1e2]
        # Transfer all relevant observations in train dataset to eval
        else:
            e1e2_rel_1[common_e1e2] = e1e2_rel_2[common_e1e2].union(e1e2_rel_1[common_e1e2])
            del e1e2_rel_2[common_e1e2]
    print('')


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

def compute_sentence_partition_balance(partition):
    subject_prop = defaultdict(lambda: 0)
    object_prop = defaultdict(lambda: 0)
    relation_prop = defaultdict(lambda: 0)
    subject_relation_prop = defaultdict(lambda: 0)
    relation_object_prop = defaultdict(lambda: 0)
    subject_object_prop = defaultdict(lambda: 0)
    triple_prop = defaultdict(lambda: 0)
    for d in partition:
        # Parse data
        subject = d['subj_type']
        object = d['obj_type']
        relation = d['relation']
        # Add Sentence Records
        subject_prop[subject] += 1 / len(partition)
        object_prop[object] += 1 / len(partition)
        relation_prop[relation] += 1 / len(partition)
        subject_relation_prop[f"({subject} | {relation})"] += 1 / len(partition)
        relation_object_prop[f"({relation} | {object})"] += 1 / len(partition)
        subject_object_prop[f"({subject} | {object})"] += 1 / len(partition)
        triple_prop[f"({subject} | {relation} | {object})"] += 1 / len(partition)
    return {'subject': subject_prop,
            'object': object_prop,
            'relation': relation_prop,
            'subject_relation': subject_relation_prop,
            'relation_object': relation_object_prop,
            'subject_object': subject_object_prop,
            'triple': triple_prop}

def compute_graph_partition_balance(graph):
    subject_prop = defaultdict(lambda: 0)
    object_prop = defaultdict(lambda: 0)
    relation_prop = defaultdict(lambda: 0)
    subject_relation_prop = defaultdict(lambda: 0)
    relation_object_prop = defaultdict(lambda: 0)
    subject_object_prop = defaultdict(lambda: 0)
    triple_prop = defaultdict(lambda: 0)
    for triple in graph:
        subject, relation, object = triple
        # Add Sentence Records
        subject_prop[subject] += 1 / len(graph)
        object_prop[object] += 1 / len(graph)
        relation_prop[relation] += 1 / len(graph)
        subject_relation_prop[f"({subject} | {relation})"] += 1 / len(graph)
        relation_object_prop[f"({relation} | {object})"] += 1 / len(graph)
        subject_object_prop[f"({subject} | {object})"] += 1 / len(graph)
        triple_prop[f"({subject} | {relation} | {object})"] += 1 / len(graph)

    return {'subject': subject_prop,
            'object': object_prop,
            'relation': relation_prop,
            'subject_relation': subject_relation_prop,
            'relation_object': relation_object_prop,
            'subject_object': subject_object_prop,
            'triple': triple_prop}

def group_balances(partition_dict, type='sentence'):
    # Schema:
    #   'Sentence':
    #       Subject:
    #           subjects: proportions
    schema = defaultdict(                   # Sentence/KG
                lambda: defaultdict(        # Subject/Object/Relation/etc...
                    lambda: None))          # Array of [Element, Train Val, Dev Val, Test Val, Full Val]
    # Choose arbitrary dict to iterate through
    iter_dict = partition_dict['train_ic']
    # Subject/object/relation/etc...
    for component in iter_dict:
        elements = iter_dict[component]
        # SUBJ-/OBJ-/Rel/(SUBJ-, rel)/etc...
        component_grouped = []
        for element in elements:
            train_value = round(partition_dict['train_ic'][component][element], 3)
            dev_value = round(partition_dict['dev_ic'][component][element], 3)
            test_value = round(partition_dict['test_ic'][component][element], 3)
            full_value = round(partition_dict['full'][component][element], 3)
            element_grouped = np.array([element, train_value, dev_value, test_value, full_value])
            component_grouped.append(element_grouped)
        component_grouped = np.stack(component_grouped, axis=0)
        schema[type][component] = component_grouped
        file_name = os.path.join(os.getcwd(), '{}_{}.csv'.format(type, component))
        np.savetxt(file_name, component_grouped, fmt='%s', delimiter=',')

    return schema

def read_tacred_partition(file_path):
    with open(file_path, 'r') as json_handle:
        data = json.load(json_handle)
    triples = set()
    for d in data:
        triple = (d['subj_type'], d['relation'], d['obj_type'])
        triples.add(triple)
    return data, triples

def run_obtain_partition_balances():
    tacred_dir = '/Volumes/External HDD/dataset/tacred/data/json'
    filenames = ['train_ic', 'dev_ic', 'test_ic']
    full_data = []
    full_graph = set()
    sentence_balances = {}
    graph_balances = {}
    for filename in filenames:
        file_path = os.path.join(tacred_dir, filename + '.json')
        file_data, file_graph = read_tacred_partition(file_path)
        full_data.append(file_data)
        full_graph = full_graph.union(file_graph)
        sentence_balance = compute_sentence_partition_balance(file_data)
        graph_balance = compute_graph_partition_balance(file_graph)
        sentence_balances[filename] = sentence_balance
        graph_balances[filename] = graph_balance
        print('Finished file: {}'.format(filename))

    full_data = np.concatenate(full_data, axis=0)
    sentence_balance = compute_sentence_partition_balance(full_data)
    graph_balance = compute_graph_partition_balance(full_graph)
    sentence_balances['full'] = sentence_balance
    graph_balances['full'] = graph_balance
    group_balances(sentence_balances, type='sentence')
    group_balances(graph_balances, type='graph')
    print('Finished...')

def run_create_new_TACRED():
    tacred_dir = '/Volumes/External HDD/dataset/tacred/data/json'
    e1e2_split = .2
    e1rel_split = .2
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
    # run_create_new_TACRED()
    run_obtain_partition_balances()


