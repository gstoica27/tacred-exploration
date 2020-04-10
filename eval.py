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
import json


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='Directory of the model.',
                    default='/usr0/home/gis/research/tacred-exploration/saved_models/PA-LSTM-TACRED')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='/usr0/home/gis/data/tacred/data/json')
parser.add_argument('--vocab_dir', type=str, default='/usr0/home/gis/data/tacred/data/vocab')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--test_save_dir', type=str,
                    default='/usr0/home/gis/research/tacred-exploration/tacred_test_performances',
                    help='Test save directory')

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
model = RelationModel(opt)
model.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
print('config vocab size: {} | actual size: {}'.format(
    opt['vocab_size'], vocab.size
))
print('config: {}'.format(opt))
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
data = []
correct_predictions = []
incorrect_predictions = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b)
    predictions += preds
    all_probs += probs

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

is_correct = np.array(predictions) == np.array(batch.gold())
is_incorrect = np.array(predictions) != np.array(batch.gold())
correct_predictions = np.arange(len(predictions))[is_correct]
incorrect_predictions = np.arange(len(predictions))[is_incorrect]
with open(data_file, 'rb') as handle:
    data = np.array(json.load(handle))
correct_data = data[correct_predictions].tolist()
incorrect_data = data[incorrect_predictions].tolist()

save_dir = os.path.join(opt['test_save_dir'], opt['id'], 'correctness')
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, 'correct_data.pkl'), 'wb') as handle:
    pickle.dump(correct_data, handle)
with open(os.path.join(save_dir, 'incorrect_data.pkl'), 'wb') as handle:
    pickle.dump(incorrect_data, handle)


# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

