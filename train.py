"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import pickle
import yaml
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from data.loader import DataLoader, map_to_ids
from utils.kg_vocab import KGVocab
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab
from collections import defaultdict
from configs.dict_with_attributes import AttributeDict


def extract_binary_probs(data, model):
    # id2label = dict([(v, k) for k, v in data.binary_rel2id.items()])
    predictions = []
    labels = []
    for i, batch in enumerate(data):
        _, probs, _, batch_labels = model.predict(batch)
        # pred_labels = [id2label[pred] for pred in preds]
        predictions += probs
        labels += batch_labels
    return np.array(predictions), np.array(labels)

def extract_binary_predictions(data, model, threshold=None, metric='accuracy'):
    id2rel = dict([(v, k) for k, v in data.binary_rel2id.items()])
    binary_probs, binary_labels = extract_binary_probs(data=data, model=model)
    if threshold is None:
        gold_labels = data.gold()
        binary_gold_labels = ['has_relation' if label != 'no_relation' else label for label in gold_labels]
        gold_ids = np.array([data.binary_rel2id[label] for label in binary_gold_labels])
        acc, threshold = helper.find_threshold(probs=binary_probs, true_labels=gold_ids, metric=metric)
    binary_predictions = (binary_probs.reshape(-1) > threshold).astype(np.int).tolist()
    binary_labels = np.array([id2rel[prediction] for prediction in binary_predictions])
    return binary_labels, threshold


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
train_batch = DataLoader(opt['data_dir'] + '/train.json',
                         opt['batch_size'],
                         opt,
                         vocab,
                         evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json',
                       opt['batch_size'],
                       opt,
                       vocab,
                       evaluation=True,
                       rel_graph=train_batch.e1e2_to_rel,
                       rel2id=train_batch.rel2id,
                       binary_rel2id=train_batch.binary_rel2id)
test_batch = DataLoader(opt['data_dir'] + '/test.json',
                        opt['batch_size'],
                        opt,
                        vocab,
                        evaluation=True,
                        rel_graph=train_batch.e1e2_to_rel,
                        rel2id=train_batch.rel2id,
                        binary_rel2id=train_batch.binary_rel2id)
# Get mappings
opt['rel2id'] = train_batch.rel2id
opt['no_relation_id'] = train_batch.rel2id['no_relation']
opt['binary_rel2id'] = train_batch.binary_rel2id

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

# if opt['typed_relations']:
id2label = dict([(v,k) for k,v in train_batch.rel2id.items()])
# Remove no_relation from decoder
if opt['one_vs_many']:
    opt['num_class'] = len(id2label) - 1
else:
    opt['num_class'] = len(id2label)
# model
if opt['binary_classification']:
    opt['apply_binary_classification'] = True
    binary_model = RelationModel(opt, emb_matrix=emb_matrix)

opt['apply_binary_classification'] = False
model = RelationModel(opt, emb_matrix=emb_matrix)

dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']
best_dev_metrics = defaultdict(lambda: -np.inf)
test_metrics_at_best_dev = defaultdict(lambda: -np.inf)
eval_metric = opt['eval_metric']
# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    binary_train_loss = 0
    for i, batch in enumerate(train_batch):
    # for i in range(0):
        start_time = time.time()
        global_step += 1
        # Update binary relation model
        if opt['binary_classification']:
            binary_batch = deepcopy(batch)
            binary_losses = binary_model.update(binary_batch)
            binary_train_loss += binary_losses['cumulative']

        # Update full relation model
        losses = model.update(batch)
        train_loss += losses['cumulative']

        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print_info = format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], duration, current_lr)
            loss_prints = ''
            for loss_type, loss in losses.items():
                loss_prints += ', {}: {:.6f}'.format(loss_type, loss)
            if opt['binary_classification']:
                loss_prints += ' | Binary Losses'
                for loss_type, loss in binary_losses.items():
                    loss_prints += ', {}: {:.6f}'.format(loss_type, loss)
            print(print_info + loss_prints)

    print("Evaluating on train set...")
    predictions = []
    train_probs = []
    train_eval_loss = 0
    for i, batch in enumerate(train_batch):
    # for i, _ in enumerate([]):
        preds, probs, loss, _ = model.predict(batch)
        predictions += preds
        train_eval_loss += loss
        train_probs += probs

    train_probs = np.array(train_probs)
    if opt['one_vs_many']:
        predictions, _ = helper.compute_one_vs_many_predictions(probs=train_probs,
                                                                true_label_names=train_batch.gold(),
                                                                rel2id=train_batch.rel2id,
                                                                threshold_metric=opt['threshold_metric'])

    predictions = np.array([id2label[p] for p in predictions])
    if opt['binary_classification']:
        binary_predictions, _ = extract_binary_predictions(data=train_batch,
                                                           model=binary_model,
                                                           threshold=None,
                                                           metric=opt['threshold_metric'])
        gold_labels = train_batch.gold()
        binary_gold_labels = np.array(['has_relation' if label != 'no_relation' else label for label in gold_labels])
        train_acc = sum(binary_predictions == binary_gold_labels) / len(binary_gold_labels)
        predictions[binary_predictions == 'no_relation'] = 'no_relation'
        # Accuracy over no_relation data
        no_rel_acc = sum((binary_predictions == binary_gold_labels)[binary_gold_labels == 'no_relation'])
        no_rel_acc /= sum(binary_gold_labels == 'no_relation')
        # Accuracy over relation data
        rel_acc = sum((binary_predictions == binary_gold_labels)[binary_gold_labels != 'no_relation'])
        rel_acc /= sum(binary_gold_labels != 'no_relation')
        print('Train Accuracy: {} | No_relation: {} | relation: {}'.format(train_acc, no_rel_acc, rel_acc))

    train_p, train_r, train_f1 = scorer.score(train_batch.gold(), predictions)

    train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
    train_eval_loss = train_eval_loss / train_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,
                                                                                     train_loss,
                                                                                     train_eval_loss, train_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, train_eval_loss, train_f1))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_probs = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, probs, loss, _ = model.predict(batch)
        predictions += preds
        dev_loss += loss
        dev_probs += probs

    dev_probs = np.array(dev_probs)
    if opt['one_vs_many']:
        predictions, dev_thresholds = helper.compute_one_vs_many_predictions(
            probs=dev_probs, true_label_names=dev_batch.gold(),
            rel2id=train_batch.rel2id, threshold_metric=opt['threshold_metric']
        )

    dev_predictions = np.array([id2label[p] for p in predictions])

    if opt['binary_classification']:
        binary_predictions, dev_threshold = extract_binary_predictions(data=dev_batch,
                                                                       model=binary_model,
                                                                       threshold=None,
                                                                       metric=opt['threshold_metric'])
        gold_labels = dev_batch.gold()
        binary_gold_labels = np.array(['has_relation' if label != 'no_relation' else label for label in gold_labels])
        dev_acc = sum(binary_predictions == binary_gold_labels) / len(binary_gold_labels)
        dev_predictions[binary_predictions == 'no_relation'] = 'no_relation'
        # Accuracy over no_relation data
        no_rel_acc = sum((binary_predictions == binary_gold_labels)[binary_gold_labels == 'no_relation'])
        no_rel_acc /= sum(binary_gold_labels == 'no_relation')
        # Accuracy over relation data
        rel_acc = sum((binary_predictions == binary_gold_labels)[binary_gold_labels != 'no_relation'])
        rel_acc /= sum(binary_gold_labels != 'no_relation')
        print('Dev Accuracy: {} | No_relation: {} | relation: {}'.format(dev_acc, no_rel_acc, rel_acc))
    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), dev_predictions)

    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(
        epoch, train_loss, dev_loss, dev_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

    current_dev_metrics = {'f1': dev_f1, 'precision': dev_p, 'recall': dev_r, 'acc': dev_acc}
    if opt['one_vs_many']:
        current_dev_metrics['threshold'] = dev_thresholds
    elif opt['binary_classification']:
        current_dev_metrics['threshold'] = dev_threshold

    print("Evaluating on test set...")
    predictions = []
    test_loss = 0
    test_probs = []
    for i, batch in enumerate(test_batch):
        preds, probs, loss, _ = model.predict(batch)
        predictions += preds
        test_loss += loss
        test_probs += probs

    test_probs = np.array(test_probs)
    if opt['one_vs_many']:
        predictions, _ = helper.compute_one_vs_many_predictions(probs=test_probs,
                                                             true_label_names=test_batch.gold(),
                                                             rel2id=train_batch.rel2id,
                                                             thresholds=dev_thresholds,
                                                                threshold_metric=opt['threshold_metric'])

    test_predictions = np.array([id2label[p] for p in predictions])

    if opt['binary_classification']:
        binary_predictions, _ = extract_binary_predictions(data=test_batch,
                                                           model=binary_model,
                                                           threshold=dev_threshold,
                                                           metric=opt['threshold_metric'])
        gold_labels = test_batch.gold()
        binary_gold_labels = np.array(['has_relation' if label != 'no_relation' else label for label in gold_labels])
        test_acc = sum(binary_predictions == binary_gold_labels) / len(binary_gold_labels)
        test_predictions[binary_predictions == 'no_relation'] = 'no_relation'
        # Accuracy over no_relation data
        no_rel_acc = sum((binary_predictions == binary_gold_labels)[binary_gold_labels == 'no_relation'])
        no_rel_acc /= sum(binary_gold_labels == 'no_relation')
        # Accuracy over relation data
        rel_acc = sum((binary_predictions == binary_gold_labels)[binary_gold_labels != 'no_relation'])
        rel_acc /= sum(binary_gold_labels != 'no_relation')
        print('Test Accuracy: {} | No_relation: {} | relation: {}'.format(test_acc, no_rel_acc, rel_acc))

    test_p, test_r, test_f1 = scorer.score(test_batch.gold(), test_predictions)

    test_metrics_at_current_dev = {'f1': test_f1, 'precision': test_p, 'recall': test_r, 'acc': test_acc}

    if best_dev_metrics[eval_metric] <= current_dev_metrics[eval_metric]:
        best_dev_metrics = current_dev_metrics
        test_metrics_at_best_dev = test_metrics_at_current_dev
        # Compute Confusion Matrices over triples excluded in Training
        test_triple_preds = np.array(test_predictions)
        test_triple_gold = np.array(test_batch.gold())
        dev_triple_preds = np.array(dev_predictions)
        dev_triple_gold = np.array(dev_batch.gold())
        test_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=test_triple_gold,
                                                                  predictions=test_triple_preds)
        dev_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=dev_triple_gold,
                                                                 predictions=dev_triple_preds)
        print("Saving test info...")
        with open(test_save_file, 'wb') as outfile:
            pickle.dump(test_probs, outfile)
        with open(test_confusion_save_file, 'wb') as handle:
            pickle.dump(test_confusion_matrix, handle)
        with open(dev_confusion_save_file, 'wb') as handle:
            pickle.dump(dev_confusion_matrix, handle)
    print_str = 'Best Dev Metrics |'
    for name, value in best_dev_metrics.items():
        print_str += ' {}: {} |'.format(name, value)
    print(print_str)
    print_str = 'Test Metrics at Best Dev |'
    for name, value in test_metrics_at_best_dev.items():
        print_str += ' {}: {} |'.format(name, value)
    print(print_str)

    # print("Best Dev Metrics | F1: {} | Precision: {} | Recall: {}".format(
    #     best_dev_metrics['f1'], best_dev_metrics['precision'], best_dev_metrics['recall']
    # ))
    # print("Test Metrics at Best Dev | F1: {} | Precision: {} | Recall: {}".format(
    #     test_metrics_at_best_dev['f1'], test_metrics_at_best_dev['precision'], test_metrics_at_best_dev['recall']
    # ))

    train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
    print("epoch {}: test_loss = {:.6f}, test_f1 = {:.4f}".format(epoch, test_loss, test_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, test_loss, test_f1))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
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

print("Training ended with {} epochs.".format(epoch))

