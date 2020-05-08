"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
from copy import deepcopy

from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import numpy as np
from data.process_data import DataProcessor

server_load_dir = '/usr0/home/gis/research/tacred-exploration/saved_models'
server_data_dir = '/usr0/home/gis/data/tacred/data/json'
server_vocab_dir = '/usr0/home/gis/data/tacred/data/vocab'

local_load_dir = '/Volumes/External HDD/dataset/tacred/saved_models'
local_data_dir = '/Volumes/External HDD/dataset/tacred/data/json'
local_vocab_dir = '/Volumes/External HDD/dataset/tacred/data/vocab'

cwd = os.getcwd()
on_server = 'Desktop' not in cwd
base_load_dir = server_load_dir if on_server else local_load_dir
vocab_dir = server_vocab_dir if on_server else local_vocab_dir
data_dir = server_data_dir if on_server else local_data_dir

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default=data_dir)
parser.add_argument('--vocab_dir', type=str,
                    default=vocab_dir)
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--binary_model_file', type=str,
                    default=os.path.join(base_load_dir, 'PA-LSTM-Binary'),
                    )
parser.add_argument('--positive_model_file', type=str,
                    default=os.path.join(base_load_dir, 'PA-LSTM-Full-TACRED_Positives')
                    )
parser.add_argument('--negative_model_file', type=str,
                    default=os.path.join(base_load_dir, 'PA-LSTM-Full-TACRED_Negatives'))

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

def load_model(model_file, is_binary=False, num_class=42):
    print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)
    opt['apply_binary_classification'] = is_binary
    opt['cuda'] = torch.cuda.is_available()
    opt['num_class'] = num_class
    model = RelationModel(opt)
    model.load(model_file)
    model.opt['cuda'] = torch.cuda.is_available()
    return model, opt

# Load binary model
binary_model_file = os.path.join(args.binary_model_file, args.model)
binary_model, binary_opt = load_model(model_file=binary_model_file, is_binary=True, num_class=1)
# Load positive model
positive_model_file = os.path.join(args.positive_model_file, args.model)
positive_model, positive_opt = load_model(model_file=positive_model_file, is_binary=False, num_class=42)
# Load negative model
negative_model_file = os.path.join(args.negative_model_file, args.model)
negative_model, negative_opt = load_model(model_file=negative_model_file, is_binary=False, num_class=42)
# load vocab
vocab_file = args.vocab_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
print('config vocab size: {} | actual size: {}'.format(
    binary_opt['vocab_size'], vocab.size
))
# print('Binary config: {}'.format(binary_opt))
# print('Positive config: {}'.format(positive_opt))
assert binary_opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
# TODO: May be necessary to save and load the mappings for each data iterator from the training.
#  I don't think this is neeeded atm because the data is loaded in exactly the same way between
#  the eval and train files --> same dictionary mapping values.
# load data
data_processor = DataProcessor(config=binary_opt,
                               vocab=vocab,
                               data_dir = data_dir,
                               partition_names=['train', 'dev', 'test'])

train_iterator = data_processor.create_iterator(
    config={
        'binary_classification': False,
        'exclude_negative_data': False,
        'relation_masking': False,
        'word_dropout': binary_opt['word_dropout']
    },
    partition_name='train'
)

dev_iterator =  data_processor.create_iterator(
config={
            'binary_classification': False,
            'exclude_negative_data': False,
            'relation_masking': False,
            'word_dropout': binary_opt['word_dropout']
        },
        partition_name='dev'
)
test_iterator =  data_processor.create_iterator(
config={
            'binary_classification': False,
            'exclude_negative_data': False,
            'relation_masking': False,
            'word_dropout': binary_opt['word_dropout']
        },
        partition_name='test'
)

helper.print_config(binary_opt)
helper.print_config(positive_opt)
binary_label2id = data_processor.name2id['binary_rel2id']
binary_id2label = dict([(v, k) for k, v in binary_label2id.items()])
label2id = data_processor.name2id['rel2id']
id2label = dict([(v,k) for k,v in label2id.items()])

def extract_eval_probs(dataset, model):
    data_probs = []
    for i, batch in enumerate(dataset):
        batch_probs, _, _ = model.predict(batch)
        data_probs += batch_probs
    return np.array(data_probs)

def compute_positive_accuracy(dataset, pred_probs):
    positive_probs = deepcopy(pred_probs)
    positive_probs[:, constant.LABEL_TO_ID['no_relation']] = -np.inf
    positive_preds = np.argmax(positive_probs, axis=-1)
    positive_labels = np.array([id2label[p] for p in positive_preds])
    positive_gold = [label for label in dataset.labels if label != 'no_relation']
    filtered_positives = [label for label, gold_label in zip(positive_labels, dataset.labels) if gold_label != 'no_relation']
    scorer.score(positive_gold, filtered_positives)

def evaluate_joint_models(dataset, binary_model, positive_model, negative_model, id2label, binary_id2label, threshold):
    binary_probs = extract_eval_probs(dataset=dataset, model=binary_model)
    positive_probs = extract_eval_probs(dataset=dataset, model=positive_model)
    negative_probs = extract_eval_probs(dataset=dataset, model=negative_model)

    positive_preds = np.argmax(positive_probs, axis=-1)
    positive_labels = np.array([id2label[p] for p in positive_preds])
    negative_preds = np.argmax(negative_probs, axis=-1)
    negative_labels = np.array([id2label[p] for p in negative_preds])

    binary_preds = (binary_probs > threshold).astype(int)
    print('-'*80)
    print('Binary performance...')
    print('-' * 80)
    binary_labels = np.array([binary_id2label[p] for p in binary_preds])
    binary_gold = ['has_relation' if label != 'no_relation' else label for label in dataset.labels]
    scorer.score(binary_gold, binary_labels)
    print('-' * 80)
    print('Positive Model Positive Performance:')
    print('-' * 80)
    compute_positive_accuracy(dataset, pred_probs=positive_probs)
    print('-' * 80)
    print('Positive Model Overall Performance:')
    print('-' * 80)
    scorer.score(dataset.labels, positive_labels)
    print('-' * 80)
    print('Negative Model Positive Accuracy:')
    print('-' * 80)
    compute_positive_accuracy(dataset, pred_probs=negative_probs)
    print('-' * 80)
    print("Negative Model Overall Performance")
    print('-' * 80)
    scorer.score(dataset.labels, negative_labels)
    print('-' * 80)
    print("Aggregate Performance")
    print('-' * 80)
    test_labels = []
    binary_neg_labels = []
    binary_neg_gold_labels = []
    binary_pos_labels = []
    binary_pos_gold_labels = []
    for binary_label, positive_label, negative_label, gold_label in zip(binary_labels, positive_labels, negative_labels, dataset.labels):
        if binary_label == 'no_relation':
            test_labels.append(negative_label)
            # test_labels.append(binary_label)
            binary_neg_labels.append(negative_label)
            binary_neg_gold_labels.append(gold_label)
        else:
            test_labels.append(positive_label)
            binary_pos_labels.append(positive_label)
            binary_pos_gold_labels.append(gold_label)

    metrics = scorer.score(dataset.labels, test_labels)
    print('-'*80)
    print('Positive Model Binary Positive Predictions Performance:')
    print('-' * 80)
    scorer.score(binary_pos_gold_labels, binary_pos_labels)
    print('-' * 80)
    print('Negative Model Binary Negative Predictions Performance:')
    print('-' * 80)
    scorer.score(binary_neg_gold_labels, binary_neg_labels)

    return metrics

threshold = 0.5096710324287415 # Fill this in
print('#'*80)
print('Train Dataset Performance')
print('#'*80)
evaluate_joint_models(
    dataset=train_iterator,
    binary_model=binary_model,
    positive_model=positive_model,
    negative_model=negative_model,
    id2label=id2label,
    binary_id2label=binary_id2label,
    threshold=threshold
)
print('#'*80)
print('Dev Dataset Performance')
print('#'*80)
evaluate_joint_models(
    dataset=dev_iterator,
    binary_model=binary_model,
    positive_model=positive_model,
    negative_model=negative_model,
    id2label=id2label,
    binary_id2label=binary_id2label,
    threshold=threshold
)
print('#'*80)
print('Test Dataset Performance')
print('#'*80)
evaluate_joint_models(
    dataset=test_iterator,
    binary_model=binary_model,
    positive_model=positive_model,
    negative_model=negative_model,
    id2label=id2label,
    binary_id2label=binary_id2label,
    threshold=threshold
)

print("Evaluation ended.")

