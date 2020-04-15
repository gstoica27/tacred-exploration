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

from data.loader import DataLoader, map_to_ids
from utils.kg_vocab import KGVocab
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab
from collections import defaultdict
from configs.dict_with_attributes import AttributeDict


def str2bool(v):
    return v.lower() in ('true')

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
# config_path = '/Users/georgestoica/Desktop/Research/tacred-exploration/configs/model_config.yaml'
# config_path = '/zfsauton3/home/gis/research/tacred-exploration/configs/model_config_server.yaml'
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
train_batch = DataLoader(opt['data_dir'] + '/train_wrong.json',
                         opt['batch_size'],
                         opt,
                         vocab,
                         evaluation=False)
test_batch = DataLoader(opt['data_dir'] + '/test_wrong.json',
                        opt['batch_size'],
                        opt,
                        vocab,
                        evaluation=True,
                        rel_graph=train_batch.e1e2_to_rel,
                        rel2id=train_batch.rel2id)
# Get mappings
opt['rel2id'] = train_batch.rel2id
#print(train_batch.rel2id)

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
# print model info
helper.print_config(opt)

# if opt['typed_relations']:
id2label = dict([(v,k) for k,v in train_batch.rel2id.items()])

opt['num_class'] = len(id2label)

# model
model = RelationModel(opt, emb_matrix=emb_matrix)

test_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']
best_test_metrics = defaultdict(lambda: -np.inf)
eval_metric = opt['eval_metric']
# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
    # for i in range(0):
        start_time = time.time()
        global_step += 1
        losses = model.update(batch)
        train_loss += losses['cumulative']
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print_info = format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], duration, current_lr)
            loss_prints = ''
            for loss_type, loss in losses.items():
                loss_prints += ', {}: {:.6f}'.format(loss_type, loss)
            print(print_info + loss_prints)

    print("Evaluating on train set...")
    predictions = []
    train_probs = []
    train_eval_loss = 0
    for i, batch in enumerate(train_batch):
    # for i, _ in enumerate([]):
        preds, probs, loss = model.predict(batch)
        predictions += preds
        train_eval_loss += loss
        train_probs += probs

    train_probs = np.array(train_probs)

    predictions = [id2label[p] for p in predictions]
    train_p, train_r, train_f1 = scorer.score(train_batch.gold(), predictions)

    train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
    train_eval_loss = train_eval_loss / train_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,
                                                                                     train_loss,
                                                                                     train_eval_loss, train_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, train_eval_loss, train_f1))
    total_correct = sum(np.array(predictions) == np.array(train_batch.gold()))
    train_acc = total_correct / len(predictions)
    print('Train Accuracy: {}'.format(train_acc))

    print("Evaluating on test set...")
    predictions = []
    test_loss = 0
    test_probs = []
    for i, batch in enumerate(test_batch):
        preds, probs, loss = model.predict(batch)
        predictions += preds
        test_loss += loss
        test_probs += probs

    test_probs = np.array(test_probs)
    test_predictions = [id2label[p] for p in predictions]
    test_p, test_r, test_f1 = scorer.score(test_batch.gold(), test_predictions)

    total_correct = sum(np.array(test_predictions) == np.array(test_batch.gold()))
    test_acc = total_correct / len(test_predictions)
    print('Test Accuracy: {}'.format(test_acc))
    test_metrics_at_current_dev = {'f1': test_f1, 'precision': test_p, 'recall': test_r, 'acc': test_acc}

    if best_test_metrics[eval_metric] <= test_metrics_at_current_dev[eval_metric]:
        best_test_metrics = test_metrics_at_current_dev
        # Compute Confusion Matrices over triples excluded in Training
        test_triple_preds = np.array(test_predictions)
        test_triple_gold = np.array(test_batch.gold())
        test_confusion_matrix = scorer.compute_confusion_matrices(ground_truth=test_triple_gold,
                                                                  predictions=test_triple_preds)
        print("Saving test info...")
        with open(test_save_file, 'wb') as outfile:
            pickle.dump(test_probs, outfile)
        with open(test_confusion_save_file, 'wb') as handle:
            pickle.dump(test_confusion_matrix, handle)
    print_str = 'Test Metrics at Best Dev |'
    for name, value in best_test_metrics.items():
        print_str += ' {}: {} |'.format(name, value)
    print(print_str)

    train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
    print("epoch {}: test_loss = {:.6f}, test_f1 = {:.4f}".format(epoch, test_loss, test_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, test_loss, test_f1))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or test_f1 > max(test_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    
    # lr schedule
    if len(test_f1_history) > 10 and test_f1 <= test_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    test_f1_history += [test_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))

