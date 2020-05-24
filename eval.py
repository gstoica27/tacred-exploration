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
import numpy as np
from collections import defaultdict
from utils.visualize_features import plot_histogram

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='Directory of the model.',
                    default='/Users/georgestoica/Desktop/Research/tacred-exploration/saved_models/PA-LSTM-Baseline')
parser.add_argument('--model', type=str, default='checkpoint_epoch_50.pt', help='Name of the model file.')
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

def evaluate_predictions(predicted_probs):
    predicted_probs = np.array(predicted_probs)
    # no_relation_probs = np.max((1 - predicted_probs) * np.ceil(predicted_probs), axis=1)
    no_relation_probs = np.prod((1 - predicted_probs), axis=1)
    no_relations = np.ones(predicted_probs.shape[0]) * 41
    best_relation = np.argmax(predicted_probs, axis=1)
    best_probs = np.max(predicted_probs, axis=1)
    replace_preds = no_relation_probs > best_probs
    best_relation[replace_preds] = no_relations[replace_preds]
    return best_relation

def get_num_rels2probs(probs, data, pair2rels, vocab, are_fp, are_fn):
    num_rels2data = defaultdict(lambda: defaultdict(lambda: list()))
    for d, prob,  is_fp, is_fn, in zip(data, probs, are_fp, are_fn):
        subj_type = 'SUBJ-' + d['subj_type']
        obj_type = 'OBJ-' + d['obj_type']
        subject, object = vocab.word2id[subj_type], vocab.word2id[obj_type]
        num_rels = len(pair2rels[(subject, object)])
        if is_fp:
            num_rels2data[num_rels]['fp'].append(prob)
        elif is_fn:
            num_rels2data[num_rels]['fn'].append(prob)

    for num_rels, fp_fn2probs in num_rels2data.items():
        for mistake, prob in fp_fn2probs.items():
            prob = np.array(prob)
            no_rel_prob = np.prod(1 - prob, axis=1).mean()
            best_prob = np.max(prob, axis=1).mean()
            num_rels2data[num_rels][mistake] = {'no_relation': no_rel_prob, 'relation': best_prob}
            print(f'{num_rels} | {mistake} | no_relation: {no_rel_prob}, relation: {best_prob}')
    return num_rels2data


# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['cuda'] = torch.cuda.is_available()
opt['cpu'] = not torch.cuda.is_available()
model = RelationModel(opt)
model.load(model_file)
model.opt['cuda'] = torch.cuda.is_available()
model.opt['cpu'] = not torch.cuda.is_available()

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
train_batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b)
    predictions += preds
    all_probs += probs
predictions = evaluate_predictions(all_probs)
predictions = [id2label[p] for p in predictions]
metrics = scorer.score(batch.gold(), predictions, verbose=True)

predictions = np.array(predictions)
gold = np.array(batch.gold())
is_wrong = predictions != gold

all_probs = np.array(all_probs)
is_fp = (predictions != 'no_relation') * (gold == 'no_relation')
is_fn = (predictions == 'no_relation') * (gold != 'no_relation')

get_num_rels2probs(probs=all_probs, data=batch.raw_data,
                   pair2rels=train_batch.e1e2_to_rel,
                   vocab=vocab, are_fp=is_fp, are_fn=is_fn)
plot_histogram(data=batch.raw_data, are_wrong=is_wrong, pair2rels=train_batch.e1e2_to_rel, vocab=vocab)

# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

