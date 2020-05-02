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

from data.process_data import DataProcessor
from utils.kg_vocab import KGVocab
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab
from collections import defaultdict
from configs.dict_with_attributes import AttributeDict

def extract_preds(dataset, model):
    data_preds = []
    for i, batch in enumerate(dataset):
        subjects, _, objects = batch['supplemental']['triple']
        batch_probs, _ = model.predict(batch)
        batch_preds = list(np.argmax(batch_probs, axis=-1))
        pred_triples = list(zip(subjects.detach().numpy(), batch_preds, objects.detach().numpy()))

        data_preds += pred_triples
    data_preds = [dataset.triple2id_fn(triple) for triple in data_preds]
    return data_preds

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

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

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
# print model info
helper.print_config(opt)

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

# Load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
data_processor = DataProcessor(config=opt,
                               vocab=vocab,
                               data_dir = opt['data_dir'],
                               partition_names=['train', 'dev', 'test'])
# Initialize model
opt['num_class'] = data_processor.num_rel
model = RelationModel(opt, emb_matrix=emb_matrix)

best_cross_curriculum = {}
# start training
for curriculum_stage, train_length in opt['curriculum'].items():
    print('#' * 80)
    print('Starting curriculum stage: {}. Training for {} epochs'.format(curriculum_stage, train_length))
    print('#'*80)
    # Create curriculum iterators
    train_iterator = data_processor.create_iterator(
        config={
            'exclude_negative_data': False,
            'relation_masking': False,
            'word_dropout': opt['word_dropout']
        },
        partition_name='train',
        curriculum_stage=curriculum_stage
    )
    dev_iterator = data_processor.create_iterator(
        config={
            'exclude_negative_data': False,
            'relation_masking': False,
            'word_dropout': opt['word_dropout']
        },
        partition_name='dev',
        curriculum_stage=curriculum_stage
    )
    test_iterator = data_processor.create_iterator(
        config={
            'exclude_negative_data': False,
            'relation_masking': False,
            'word_dropout': opt['word_dropout']
        },
        partition_name='test',
        curriculum_stage=curriculum_stage
    )
    if curriculum_stage == 'full':
        model.specify_SCE_criterion()
    else:
        model.specify_BCE_criterion()

    global_step = 0
    global_start_time = time.time()
    format_str = '{}: step {}/{} (epoch {}/{}), ({:.3f} sec/batch), lr: {:.6f}'
    max_steps = len(train_iterator) * opt['num_epoch']
    best_dev_metrics = defaultdict(lambda: -np.inf)
    test_metrics_at_best_dev = defaultdict(lambda: -np.inf)
    eval_metric = opt['eval_metric']
    dev_f1_history = []
    current_lr = opt['lr']
    # save
    stage_model_save_dir = os.path.join(model_save_dir, 'curriculum_{}'.format(curriculum_stage))
    os.makedirs(stage_model_save_dir, exist_ok=True)
    # Dummy arg so print statement below doesn't fail
    epoch = 0
    for epoch in range(1, train_length + 1):
        train_loss = 0
        binary_train_loss = 0
        # Pass through one epoch of binary model
        print('Curriculum Stage: {} | Training Epoch: {}'.format(curriculum_stage, epoch))
        train_epoch(model=model,
                    dataset=train_iterator,
                    opt=opt,
                    global_step=global_step,
                    max_steps=max_steps,
                    epoch=epoch,
                    current_lr=current_lr)

        print('Evaluating on train set...')
        train_pred_ids = extract_preds(dataset=train_iterator, model=model)
        train_pred_labels = [train_iterator.id2label[pred_id] for pred_id in train_pred_ids]
        train_precision, train_recall, train_f1 = scorer.score(train_iterator.labels, train_pred_labels)
        train_loss = train_loss / train_iterator.num_examples * opt['batch_size']  # avg loss per batch

        print('Evaluating on dev set...')
        dev_pred_ids = extract_preds(dataset=dev_iterator, model=model)
        dev_pred_labels = [dev_iterator.id2label[pred_id] for pred_id in dev_pred_ids]
        dev_precision, dev_recall, dev_f1 = scorer.score(dev_iterator.labels, dev_pred_labels)
        print("epoch {}: train_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch, train_loss, dev_f1))
        file_logger.log("{}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_f1))

        print('Evaluating on test set...')
        test_pred_ids = extract_preds(dataset=test_iterator, model=model)
        test_pred_labels = [test_iterator.id2label[pred_id] for pred_id in test_pred_ids]
        test_precision, test_recall, test_f1 = scorer.score(test_iterator.labels, test_pred_labels)
        print("epoch {}: train_loss = {:.6f}, test_f1 = {:.4f}".format(epoch, train_loss, test_f1))
        file_logger.log("{}\t{:.6f}\t{:.4f}".format(epoch, train_loss, test_f1))
        # Record metrics
        current_dev_metrics = {'precision': dev_precision, 'recall': dev_recall, 'f1': dev_f1}
        test_metrics_at_current_dev = {'precision': test_precision, 'recall': test_recall, 'f1': test_f1}

        if best_dev_metrics[eval_metric] <= current_dev_metrics[eval_metric]:
            best_dev_metrics = current_dev_metrics
            test_metrics_at_best_dev = test_metrics_at_current_dev
            # Preprocess for confusion matrices
            dev_predictions = np.array(dev_pred_labels)
            dev_gold = np.array(dev_iterator.labels)
            test_predictions = np.array(test_pred_labels)
            test_gold = np.array(test_iterator.labels)

            dev_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=dev_gold,
                                                                     predictions=dev_predictions)
            test_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=test_gold,
                                                                      predictions=test_predictions)
            print('Saving Best Confusion Matrices')
            stage_test_save_dir = os.path.join(test_save_dir, 'curriculum_{}'.format(curriculum_stage))
            os.makedirs(stage_test_save_dir, exist_ok=True)
            test_confusion_save_file = os.path.join(stage_test_save_dir, 'test_confusion_matrix.pkl')
            dev_confusion_save_file = os.path.join(stage_test_save_dir, 'dev_confusion_matrix.pkl')
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

        model_file = os.path.join(stage_model_save_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
        model.save(model_file, epoch)

        if epoch == 1 or dev_f1 > max(dev_f1_history):
            # , model_save_dir + '/best_model.pt'
            copyfile(model_file, os.path.join(stage_model_save_dir, 'best_model.pt'))
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
    print('Loading best model from curriculum {}...'.format(curriculum_stage))
    model.load(os.path.join(stage_model_save_dir, 'best_model.pt'))
    best_cross_curriculum[curriculum_stage] = {'dev': best_dev_metrics, 'test': test_metrics_at_best_dev}

print('#'*80)
print('Performances across curriculum:')
for curriculum_stage, performances in best_cross_curriculum.items():
    print('Curriculum: {}'.format(curriculum_stage))
    for partition_name, metrics in performances.items():
        print_str = '{} performances | '.format(partition_name)
        for metric_name, metric_value in metrics.items():
            print_str += '{}: {:.4f}, '.format(metric_name, metric_value)
        print(print_str)