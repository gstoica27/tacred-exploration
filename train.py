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

from data.loader import DataLoader
from utils.kg_vocab import KGVocab
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab
from collections import defaultdict
from configs.dict_with_attributes import AttributeDict


def str2bool(v):
    return v.lower() in ('true')

def add_kg_model_params(cfg_dict):
    fact_checking_config = os.path.join(cwd, 'configs', 'fact_checking_configs.yaml')
    with open(fact_checking_config, 'r') as file:
        fact_checking_config_dict = yaml.load(file)
    fact_checking_model = cfg_dict['kg_loss']['model']
    params = fact_checking_config_dict[fact_checking_model]
    params['name'] = fact_checking_model
    return params

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

if cfg_dict['kg_loss'] is not None:
    kg_vocab = KGVocab(cfg_dict['kg_loss']['vocab_path'])
else:
    kg_vocab = None

add_encoding_config(cfg_dict)
if cfg_dict['kg_loss'] is not None:
    cfg_dict['kg_loss']['model'] = add_kg_model_params(cfg_dict)
    cfg_dict['kg_loss']['model']['num_entities'] = kg_vocab.return_num_ent()
    cfg_dict['kg_loss']['model']['num_relations'] = kg_vocab.return_num_rel()
    cfg_dict['kg_loss']['model']['freeze_embeddings'] = cfg_dict['kg_loss']['freeze_embeddings']

print(cfg_dict)
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
opt['num_class'] = len(constant.LABEL_TO_ID)

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
# opt['kg_e2_idxs'] = opt['subj_idxs'] + opt['obj_idxs']

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'],
                         opt, vocab, evaluation=False, kg_vocab=kg_vocab)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'],
                       opt, vocab, evaluation=True, kg_vocab=kg_vocab,
                       kg_graph=train_batch.kg_graph, rel_graph=train_batch.e1e2_to_rel)
test_batch = DataLoader(opt['data_dir'] + '/test.json', opt['batch_size'],
                        opt, vocab, evaluation=True, kg_vocab=kg_vocab,
                        kg_graph=train_batch.kg_graph, rel_graph=train_batch.e1e2_to_rel)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = os.path.join(opt['save_dir'], model_id)
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                    header="# epoch\ttrain_loss\tdev_loss\tdev_f1")


test_save_dir = os.path.join(opt['test_save_dir'], opt['id'])
os.makedirs(test_save_dir, exist_ok=True)
test_save_file = os.path.join(test_save_dir, 'test_records.pkl')

# print model info
helper.print_config(opt)

# model
model = RelationModel(opt, emb_matrix=emb_matrix)

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']
best_dev_metrics = defaultdict(lambda: -np.inf)
test_metrics_at_best_dev = defaultdict(lambda: -np.inf)

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
    train_eval_loss = 0
    for i, batch in enumerate(train_batch):
    # for i, _ in enumerate([]):
        preds, _, loss = model.predict(batch)
        predictions += preds
        train_eval_loss += loss
    predictions = [id2label[p] for p in predictions]
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
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss = model.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [id2label[p] for p in predictions]
    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
    
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
            train_loss, dev_loss, dev_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))
    current_dev_metrics = {'f1': dev_f1, 'precision': dev_p, 'recall': dev_r}

    print("Evaluating on test set...")
    predictions = []
    test_loss = 0
    test_preds = []
    for i, batch in enumerate(test_batch):
        preds, probs, loss = model.predict(batch)
        predictions += preds
        test_loss += loss
        test_preds += probs
    predictions = [id2label[p] for p in predictions]
    test_p, test_r, test_f1 = scorer.score(test_batch.gold(), predictions)
    test_metrics_at_current_dev = {'f1': test_f1, 'precision': test_p, 'recall': test_r}

    if best_dev_metrics['f1'] < current_dev_metrics['f1']:
        best_dev_metrics = current_dev_metrics
        test_metrics_at_best_dev = test_metrics_at_current_dev
        print("Saving test info...")
        with open(test_save_file, 'wb') as outfile:
            pickle.dump(test_preds, outfile)

    print("Best Dev Metrics | F1: {} | Precision: {} | Recall: {}".format(
        best_dev_metrics['f1'], best_dev_metrics['precision'], best_dev_metrics['recall']
    ))
    print("Test Metrics at Best Dev | F1: {} | Precision: {} | Recall: {}".format(
        test_metrics_at_best_dev['f1'], test_metrics_at_best_dev['precision'], test_metrics_at_best_dev['recall']
    ))

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

