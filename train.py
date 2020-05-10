"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
from shutil import copyfile
import torch
import pickle
import yaml
import json

from data.process_data import DataProcessor
from model.rnn import RelationModel
from utils import scorer, helper
from utils.vocab import Vocab
from collections import defaultdict

def save_filtered_data(data_dir, filter_idxs, filename='train_positives.json'):
    print('There are: {} elements in filter'.format(len(filter_idxs)))
    data_file = os.path.join(data_dir, 'train.json')
    with open(data_file, 'rb') as handle:
        data_data = json.load(handle)
    data_data = np.array(data_data)
    filtered_data = data_data[filter_idxs].tolist()
    filtered_file = os.path.join(data_dir, filename)
    json.dump(filtered_data, open(filtered_file, 'w'))


def extract_eval_probs(dataset, model, with_idxs=False):
    data_probs = []
    data_idxs = []
    for i, batch in enumerate(dataset):
        batch_probs, _, batch_idxs = model.predict(batch)
        data_probs += batch_probs
        data_idxs += batch_idxs
    if with_idxs:
        return np.array(data_probs), data_idxs
    else:
        return np.array(data_probs)

def assign_labels(binary_probs,
                  positive_probs,
                  binary_id2rel,
                  positive_id2rel,
                  gold_labels,
                  data_processor,
                  is_hard=True,
                  metric='accuracy',
                  threshold=None):
    binary_rel2id = data_processor.name2id['binary_rel2id']
    extra = {}
    # Hard disjoint models
    if is_hard:
        binary_gold_labels = ['has_relation' if label != 'no_relation' else label for label in gold_labels]
        if threshold is None:
            gold_ids = np.array([binary_rel2id[label] for label in binary_gold_labels])
            acc, threshold = helper.find_threshold(probs=binary_probs, true_labels=gold_ids, metric=metric)

        binary_predictions = (binary_probs.reshape(-1) > threshold).astype(np.int).tolist()
        binary_labels = np.array([binary_id2rel[prediction] for prediction in binary_predictions])
        positive_predictions = np.argmax(positive_probs, axis=1)
        positive_labels = np.array([positive_id2rel[pred] for pred in positive_predictions])
        positive_labels[binary_labels == 'no_relation'] = 'no_relation'
        labels = positive_labels
        extra['threshold'] = threshold
    else:
        binary_probs = binary_probs.reshape((-1, 1))
        joint_probs = binary_probs * positive_probs
        negative_probs = 1 - binary_probs
        all_probs = np.concatenate((joint_probs, negative_probs), axis=1)
        predictions = np.argmax(all_probs, axis=1)
        labels = [positive_id2rel[pred_id] for pred_id in predictions]
    return labels, extra

def find_binary_threshold(gold_ids, prediction_probs, threshold_metric='f1'):
    acc, threshold = helper.find_threshold(probs=prediction_probs, true_labels=gold_ids, metric=threshold_metric)
    perf_print = 'Threshold {} Performance: '
    for name, metric in acc.items():
        perf_print += '{}: {}, '.format(name, metric)
    print(perf_print)
    return threshold


def add_encoding_config(cfg_dict):
    if cfg_dict['encoding_type'] == 'BiLSTM':
        cfg_dict['encoding_dim'] = cfg_dict['hidden_dim'] * 2
        cfg_dict['bidirectional_encoding'] = True
    elif cfg_dict['encoding_type'] == 'LSTM':
        cfg_dict['encoding_dim'] = cfg_dict['hidden_dim']
        cfg_dict['bidirectional_encoding'] = False


cwd = os.getcwd()
on_server = 'Desktop' not in cwd
config_path = os.path.join(cwd, 'configs', f'model_config{"_nell" if on_server else ""}.yaml')
with open(config_path, 'r') as file:
    cfg_dict = yaml.load(file)

add_encoding_config(cfg_dict)

opt = cfg_dict#AttributeDict(cfg_dict)
opt['cuda'] = torch.cuda.is_available()
opt['cpu'] = not opt['cuda']
torch.manual_seed(opt['seed'])
np.random.seed(opt['seed'])
random.seed(opt['seed'])
if opt['cpu']:
    opt['cuda'] = False
elif opt['cuda']:
    torch.cuda.manual_seed(opt['seed'])

# opt = vars(args)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

opt['subj_idxs'] = vocab.subj_idxs
opt['obj_idxs'] = vocab.obj_idxs
# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
data_processor = DataProcessor(config=opt,
                               vocab=vocab,
                               data_dir = opt['data_dir'],
                               partition_names=['train_split-0.1', 'test_split-0.1', 'test'])
if opt['experiment_type'] == 'binary':
    config = {
        'binary_classification': True,
        'exclude_negative_data': False,
        'relation_masking': False,
        'word_dropout': opt['word_dropout']
    }
else:
    config={
        'binary_classification': False,
        'exclude_negative_data': opt['exclude_negatives'],
        'relation_masking': False,
        'word_dropout': opt['word_dropout']
    }

train_iterator = data_processor.create_iterator(
        config=config,
        partition_name='train_split-0.1'
    )
dev_iterator = data_processor.create_iterator(
    config=config,
    partition_name='test_split-0.1'
)
test_iterator = data_processor.create_iterator(
    config=config,
    partition_name='test'
)
# Get mappings
opt['id2label'] = train_iterator.id2label

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = os.path.join(opt['save_dir'], model_id)
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

print(cfg_dict)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                    header="# epoch\ttrain_loss\tdev_loss\tdev_f1")


test_save_dir = os.path.join(opt['test_save_dir'], opt['id'])
os.makedirs(test_save_dir, exist_ok=True)
test_save_file = os.path.join(test_save_dir, 'test_records.pkl')
test_confusion_save_file = os.path.join(test_save_dir, 'test_confusion_matrix.pkl')
dev_confusion_save_file = os.path.join(test_save_dir, 'dev_confusion_matrix.pkl')
# print model info
helper.print_config(opt)

# Remove no_relation from decoder
if opt['experiment_type'] == 'binary':
    opt['apply_binary_classification'] = True
else:
    opt['apply_binary_classification'] = False
    opt['num_class'] = 42
model = RelationModel(opt, emb_matrix=emb_matrix)

dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_iterator) * opt['num_epoch']
best_dev_metrics = defaultdict(lambda: -np.inf)
test_metrics_at_best_dev = defaultdict(lambda: -np.inf)
train_metrics_at_best_dev = defaultdict(lambda: -np.inf)
eval_metric = opt['eval_metric']

def train_epoch(model, dataset, opt, global_step, max_steps, epoch, current_lr):
    train_loss = 0
    for i, batch in enumerate(dataset):
        start_time = time.time()
        global_step += 1
        losses = model.update(batch)
        train_loss += losses['cumulative']

        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print_info = format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                           opt['num_epoch'], duration, current_lr)
            loss_prints = ''
            for loss_type, loss in losses.items():
                loss_prints += ', {}: {:.6f}'.format(loss_type, loss)
            print(print_info + loss_prints)

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    binary_train_loss = 0
    # Pass through one epoch of binary model
    train_epoch(model=model,
                dataset=train_iterator,
                opt=opt,
                global_step=global_step,
                max_steps=max_steps,
                epoch=epoch,
                current_lr=current_lr)

    print("Evaluating on train set...")
    train_loss = train_loss / train_iterator.num_examples * opt['batch_size']  # avg loss per batch
    # Evaluate on binary training set
    train_metrics = {}
    if opt['experiment_type'] == 'binary':
        train_binary_probs, data_idxs = extract_eval_probs(dataset=train_iterator, model=model, with_idxs=True)
        gold_ids = [train_iterator.label2id[l] for l in train_iterator.labels]
        train_threshold = find_binary_threshold(gold_ids=gold_ids, prediction_probs=train_binary_probs, threshold_metric='f1')
        train_binary_preds = (train_binary_probs >= train_threshold).astype(int)
        train_labels = [train_iterator.id2label[p] for p in train_binary_preds]
        positive_idxs = []
        negative_idxs = []
        incorrect_idxs = []
        for idx, (data_idx, pred_label) in enumerate(zip(data_idxs, train_labels)):
            if pred_label == 'has_relation':
                positive_idxs.append(data_idx)
            elif pred_label == 'no_relation':
                negative_idxs.append(data_idx)
            if train_iterator.labels[idx] != pred_label:
                incorrect_idxs.append(data_idx)

        train_metrics['positive_idxs'] = positive_idxs
        train_metrics['negative_idxs'] = negative_idxs
        train_metrics['incorrect_idxs'] = incorrect_idxs

        print('Length of positive predicted elements: {} | unique: {} | Num Incorrect: {}'.format(
            len(positive_idxs), len(np.unique(positive_idxs)), len(incorrect_idxs)))
    else:
        train_preds = np.argmax(extract_eval_probs(dataset=train_iterator, model=model), axis=1)
        train_labels = [train_iterator.id2label[p] for p in train_preds]
    train_metrics.update(scorer.score(train_iterator.labels, train_labels))

    print("epoch {}: train_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch, train_loss, train_metrics['f1']))
    file_logger.log("{}\t{:.6f}\t{:.4f}".format(epoch, train_loss, train_metrics['f1']))

    # eval on dev
    current_dev_metrics = {}
    print("Evaluating on dev set...")
    if opt['experiment_type'] == 'binary':
        dev_binary_probs = extract_eval_probs(dataset=dev_iterator, model=model)
        gold_ids = [data_processor.name2id['binary_rel2id'][l] for l in dev_iterator.labels]
        dev_threshold = find_binary_threshold(gold_ids=gold_ids, prediction_probs=dev_binary_probs, threshold_metric='f1')
        dev_binary_preds = (dev_binary_probs >= dev_threshold).astype(int)
        dev_labels = [dev_iterator.id2label[p] for p in dev_binary_preds]
        current_dev_metrics['threshold'] = dev_threshold
    else:
        dev_preds = np.argmax(extract_eval_probs(dataset=dev_iterator, model=model), axis=1)
        dev_labels = [dev_iterator.id2label[p] for p in dev_preds]
    current_dev_metrics.update(scorer.score(dev_iterator.labels, dev_labels))

    print("epoch {}: train_loss = {:.6f}, dev_f1 = {:.4f}".format(
        epoch, train_loss, current_dev_metrics['f1']))
    file_logger.log("{}\t{:.6f}\t{:.4f}".format(epoch, train_loss, current_dev_metrics['f1']))

    dev_f1 = current_dev_metrics['f1']

    print("Evaluating on test set...")
    if opt['experiment_type'] == 'binary':
        test_probs = extract_eval_probs(dataset=test_iterator, model=model)
        test_preds = (test_probs > dev_threshold).astype(int)
        test_labels = [test_iterator.id2label[p] for p in test_preds]
    else:
        test_preds = np.argmax(extract_eval_probs(dataset=test_iterator, model=model), axis=1)
        test_labels = [test_iterator.id2label[p] for p in test_preds]

    test_metrics_at_current_dev = scorer.score(test_iterator.labels, test_labels)

    print("epoch {}: train_loss = {:.6f}, test_f1 = {:.4f}".format(
        epoch, train_loss, test_metrics_at_current_dev['f1']))
    file_logger.log("{}\t{:.6f}\t{:.4f}".format(epoch, train_loss, test_metrics_at_current_dev['f1']))

    if best_dev_metrics[eval_metric] <= current_dev_metrics[eval_metric]:
        best_dev_metrics = current_dev_metrics
        test_metrics_at_best_dev = test_metrics_at_current_dev
        train_metrics_at_best_dev = train_metrics
        # Compute Confusion Matrices over triples excluded in Training
        test_preds = np.array(test_labels)
        dev_preds = np.array(dev_labels)
        if opt['experiment_type'] == 'binary':
            test_gold = np.array(test_iterator.labels)
            dev_gold = np.array(dev_iterator.labels)
        else:
            test_gold = np.array(test_iterator.labels)
            dev_gold = np.array(dev_iterator.labels)
        test_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=test_gold,
                                                                  predictions=test_preds)
        dev_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=dev_gold,
                                                                 predictions=dev_preds)
        print("Saving test info...")
        with open(test_confusion_save_file, 'wb') as handle:
            pickle.dump(test_confusion_matrix, handle)
        with open(dev_confusion_save_file, 'wb') as handle:
            pickle.dump(dev_confusion_matrix, handle)
    print_str = 'Best Dev Metrics |'
    for name, value in best_dev_metrics.items():
        if 'positive_idxs' == name: print_str += ' {}:{} |'.format(name, len(value))
        else: print_str += ' {}: {} |'.format(name, value)
    print(print_str)
    print_str = 'Test Metrics at Best Dev |'
    for name, value in test_metrics_at_best_dev.items():
        print_str += ' {}: {} |'.format(name, value)
    print(print_str)
    print_str = 'Train Metrics at Best Dev |'
    for name, value in train_metrics_at_best_dev.items():
        if 'idxs' in name:
            print_str += ' {}:{} |'.format(name, len(value))
        else:
            print_str += ' {}: {} |'.format(name, value)
    print(print_str)

    # save
    model_file = os.path.join(model_save_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
    model.save(model_file, epoch)

    if epoch == 1 or dev_f1 > max(dev_f1_history):
        # , model_save_dir + '/best_model.pt'
        copyfile(model_file, os.path.join(model_save_dir, 'best_model.pt'))
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    # lr schedule
    if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print('Filtering Training Data...')
if opt['experiment_type'] == 'binary':
   save_filtered_data(opt['data_dir'], train_metrics_at_best_dev['positive_idxs'], filename='train_positives.json')
   save_filtered_data(opt['data_dir'], train_metrics_at_best_dev['negative_idxs'], filename='train_negatives.json')
print("Training ended with {} epochs.".format(epoch))

