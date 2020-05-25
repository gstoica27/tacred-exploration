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
from utils.visualize_features import plot_histogram
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='Directory of the model.',
                    default='/Users/georgestoica/Desktop/Research/tacred-exploration/saved_models/PA-LSTM')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='/Volumes/External HDD/dataset/tacred/data/json')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

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

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
with_cuda = torch.cuda.is_available()
opt['cuda'] = with_cuda
opt['cpu'] = not with_cuda
model = RelationModel(opt)
model.load(model_file)
model.opt['cuda'] = with_cuda
model.opt['cpu'] = not with_cuda

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
print('config vocab size: {} | actual size: {}'.format(
    opt['vocab_size'], vocab.size
))
print('config: {}'.format(opt))
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = args.data_dir + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
train_batch = DataLoader(os.path.join(args.data_dir, 'train.json'), opt['batch_size'], opt, vocab, evaluation=True)
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True, rel_graph=train_batch.e1e2_to_rel)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b)
    predictions += preds
    all_probs += probs
predictions = [id2label[p] for p in predictions]
metrics = scorer.score(batch.gold(), predictions, verbose=True)

bracket2preds = defaultdict(lambda: list())
bracket2labels = defaultdict(lambda: list())

for (sample, pred, label) in zip(batch.raw_data, predictions, batch.gold()):
    subj_type = 'SUBJ-' + sample['subj_type']
    obj_type = 'OBJ-' + sample['obj_type']
    subject, object = vocab.word2id[subj_type], vocab.word2id[obj_type] - 4
    num_rels = len(train_batch.e1e2_to_rel[(subject, object)])
    bracket2preds[num_rels].append(pred)
    bracket2labels[num_rels].append(label)

for bracket in bracket2preds.keys():
    bracket_preds = bracket2preds[bracket]
    bracket_labels = bracket2labels[bracket]
    print(f'Performance for bracket {bracket}')
    scorer.score(bracket_labels, bracket_preds, verbose=False)

predictions = np.array(predictions)
gold = np.array(batch.gold())
is_wrong = predictions != gold

all_probs = np.array(all_probs)
is_fp = (predictions != 'no_relation') * (gold == 'no_relation')
is_fn = (predictions == 'no_relation') * (gold != 'no_relation')

eval_data = np.array(batch.raw_data)
plot_histogram(data=eval_data, are_wrong=is_wrong, pair2rels=train_batch.e1e2_to_rel, vocab=vocab)


# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

