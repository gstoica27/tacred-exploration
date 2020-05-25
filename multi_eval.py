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

parser = argparse.ArgumentParser()
parser.add_argument('--models_dir', type=str, help='Directory of the model.',
                    default='/Users/georgestoica/Desktop/Research/tacred-exploration/saved_models')
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

def load_model(model_file):
    print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)
    with_cuda = torch.cuda.is_available()
    opt['cuda'] = with_cuda
    opt['cpu'] = not with_cuda
    model = RelationModel(opt)
    model.load(model_file)
    model.opt['cuda'] = with_cuda
    model.opt['cpu'] = not with_cuda
    return model, opt

def get_bracket2model(models_dir, relation_brackets):
    bracket2model = {}
    for relation_bracket in relation_brackets:
        model_name = f'PA-LSTM-{relation_bracket}Only'
        model_file = os.path.join(models_dir, model_name, 'best_model.pt')
        bracket2model[relation_bracket], opt = load_model(model_file)
    bracket2model['rest'], _ = load_model(os.path.join(models_dir, 'PA-LSTM', 'best_model.pt'))
    return bracket2model, opt

def multi_model_eval(bracket2model, data_loader):
    all_predictions = []
    gold_labels = []
    for bracket, batches in data_loader.bracket2batch.items():
        if bracket not in bracket2model:
            model = bracket2model['rest']
        else:
            model = bracket2model[bracket]
        labels = data_loader.bracket2labels[bracket]
        bracket_predictions = []
        gold_labels += labels
        for batch in batches:
            tensor_batch = data_loader.ready_data_batch(batch)
            preds, probs, _ = model.predict(tensor_batch)
            pred_labels = [id2label[p] for p in preds]
            bracket_predictions += pred_labels
            all_predictions += preds
        print(f'Performances for Bracket: {bracket}')
        scorer.score(labels, bracket_predictions, verbose=False)

    print('Overall Performance')
    predictions = [id2label[p] for p in all_predictions]
    metrics = scorer.score(gold_labels, predictions, verbose=True)
    return metrics

# load opt
# relation_brackets = [2,3,4,5,6,7,11]
relation_brackets = [2, 3, 7]
model_names = set([f'PA-LSTM-{i}Only' for i in [2,3,4,5,6,7,11]])
bracket2model, opt = get_bracket2model(models_dir=args.models_dir, relation_brackets=relation_brackets)
# load vocab
vocab_file = os.path.join(args.models_dir, 'PA-LSTM-2Only', 'vocab.pkl')
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

multi_model_eval(bracket2model=bracket2model, data_loader=batch)

# predictions = np.array(predictions)
# gold = np.array(batch.gold())
# is_wrong = predictions != gold
#
# all_probs = np.array(all_probs)
# is_fp = (predictions != 'no_relation') * (gold == 'no_relation')
# is_fn = (predictions == 'no_relation') * (gold != 'no_relation')
#
# eval_data = np.array(batch.raw_data)
# plot_histogram(data=eval_data, are_wrong=is_wrong, pair2rels=train_batch.e1e2_to_rel, vocab=vocab)


# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

