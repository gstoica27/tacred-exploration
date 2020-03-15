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
import torch.nn as nn
import torch.optim as optim

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab
from collections import defaultdict


def str2bool(v):
    return v.lower() in ('true')

cwd = "/Volumes/External HDD/"
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.path.join(cwd, 'dataset/tacred/data/json'))
parser.add_argument('--vocab_dir', type=str, default=os.path.join(cwd, 'dataset/tacred/data/vocab'))
parser.add_argument('--test_save_dir', type=str, default=os.path.join(cwd, 'dataset/tacred/test_perfs'))
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
parser.add_argument('--no-attn', dest='attn', action='store_false')
parser.set_defaults(attn=True)
parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str,
                    default=os.path.join(cwd, 'dataset/tacred/saved_models'), help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--nas_rnn', type=str2bool, default=False)
parser.add_argument('--nas_mlp', type=str2bool, default=False)
parser.add_argument('--no_type', type=str2bool, default=False,
                    help='Whether to replace subj and obj granular typing or just Subj, and obj')
parser.add_argument('--avg_types', type=str2bool, default=False,
                    help='Whether to make SUBJ and OBJ embeddings average of granular types (SUBJ-*, OBJ-*)')
parser.add_argument('--use_cpg', type=str2bool, default=False,
                    help='Whether to use CPG Attention or not')
parser.add_argument('--difference_type_spaces', type=str2bool, default=False)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)
opt['num_class'] = len(constant.LABEL_TO_ID)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size - 2
assert emb_matrix.shape[1] == opt['emb_dim']

opt['subj_idxs'] = vocab.subj_idxs
opt['obj_idxs'] = vocab.obj_idxs

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True)
test_batch = DataLoader(opt['data_dir'] + '/test.json', opt['batch_size'], opt, vocab, evaluation=True)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

opt['arc_connections'] = [('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)]
opt['arc_merge_layers'] = range(1, 9)
opt['dropout_x'] = .04
opt['dropout_h'] = .25

test_save_dir = os.path.join(opt['test_save_dir'], opt['id'])
os.makedirs(test_save_dir, exist_ok=True)
test_save_file = os.path.join(test_save_dir, 'test_records.pkl')

opt['cpg'] = {
    'network_structure': [],
    'dropout': 0.,
    'use_batch_norm': True,
    'batch_norm_momentum': .1,
    'use_bias': False
}
opt['type_enc_dim'] = 5
opt['type_dim'] = 10

# print model info
helper.print_config(opt)

# model
model = RelationModel(opt, emb_matrix=emb_matrix)

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
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
        loss = model.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    print("Evaluating on train set...")
    predictions = []
    train_eval_loss = 0
    for i, batch in enumerate(train_batch):
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

