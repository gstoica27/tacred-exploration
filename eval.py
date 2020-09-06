"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import yaml
import numpy as np


def generate_param_list(params, cfg_dict, prefix=''):
    param_list = prefix
    for param in params:
        if param_list == '':
            param_list += f'{cfg_dict[param]}'
        else:
            param_list += f'-{cfg_dict[param]}'
    return param_list

def create_model_name(cfg_dict):
    top_level_name = 'TACRED-{}-{}'.format(cfg_dict['data_type'].upper(), cfg_dict['version'].upper())
    approach_type = 'CGCN-JRRELP' if cfg_dict['link_prediction'] is not None else 'CGCN'
    optim_name = ['optim', 'lr', 'lr_decay', 'conv_l2', 'pooling_l2', 'max_grad_norm', 'seed']
    base_params = ['emb_dim', 'ner_dim', 'pos_dim', 'hidden_dim', 'num_layers', 'mlp_layers',
                   'input_dropout', 'gcn_dropout', 'word_dropout', 'lower', 'prune_k', 'no_adj']

    param_name_list = [top_level_name, approach_type]

    optim_name = generate_param_list(optim_name, cfg_dict, prefix='optim')
    param_name_list.append(optim_name)

    main_name = generate_param_list(base_params, cfg_dict, prefix='base')
    param_name_list.append(main_name)

    if cfg_dict['rnn']:
        rnn_params = ['rnn_hidden', 'rnn_layers', 'rnn_dropout']
        rnn_name = generate_param_list(rnn_params, cfg_dict, prefix='rnn')
        param_name_list.append(rnn_name)

    if cfg_dict['link_prediction'] is not None:
        kglp_task_cfg = cfg_dict['link_prediction']
        jrrelp_params = ['label_smoothing', 'lambda', 'free_network',
                       'with_relu', 'without_observed',
                       'without_verification', 'without_no_relation']
        jrrelp_name = generate_param_list(jrrelp_params, kglp_task_cfg, prefix='jrrelp')
        param_name_list.append(jrrelp_name)

        kglp_params = ['input_drop', 'hidden_drop', 'feat_drop', 'rel_emb_dim', 'use_bias', 'filter_channels', 'stride']
        lp_cfg = cfg_dict['link_prediction']['model']
        kglp_name = generate_param_list(kglp_params, lp_cfg, prefix='kglp')
        param_name_list.append(kglp_name)

    aggregate_name = os.path.join(*param_name_list)
    return aggregate_name

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

cwd = os.getcwd()
on_server = 'Desktop' not in cwd
config_path = os.path.join(cwd, 'configs', f'{"server" if on_server else "local"}_config.yaml')

def add_kg_model_params(cfg_dict, cwd):
    link_prediction_cfg_file = os.path.join(cwd, 'configs', 'link_prediction_configs.yaml')
    with open(link_prediction_cfg_file, 'r') as handle:
        link_prediction_config = yaml.load(handle)
    link_prediction_model = cfg_dict['link_prediction']['model']
    params = link_prediction_config[link_prediction_model]
    params['name'] = link_prediction_model
    params['freeze_network'] = cfg_dict['link_prediction']['freeze_network']
    return params

with open(config_path, 'r') as file:
    cfg_dict = yaml.load(file)

opt = cfg_dict
#opt = vars(args)
torch.manual_seed(opt['seed'])
np.random.seed(opt['seed'])
random.seed(1234)
if opt['cpu']:
    opt['cuda'] = False
elif opt['cuda']:
    torch.cuda.manual_seed(opt['seed'])

# load opt
model_load_dir = opt['save_dir'] + '/' + opt['id']
print(model_load_dir)
model_file = os.path.join(model_load_dir, 'best_model.pt')
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = RelationModel(opt)
model.load(model_file)
model_load_dir = opt['save_dir'] + '/' + opt['id']
print(model_load_dir)
# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
# Add subject/object indices
opt['object_indices'] = vocab.obj_idxs
# load data
data_file = opt['data_dir'] +f'/{opt["data_type"]}/test_{opt["version"]}.json'
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b)
    predictions += preds
    all_probs += probs
predictions = [id2label[p] for p in predictions]
scorer.score(batch.gold(), predictions, verbose=True)

# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

