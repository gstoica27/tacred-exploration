"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch

from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import numpy as np
from data.process_data import DataProcessor

server_load_dir = '/usr0/home/gis/research/tacred-exploration/saved_models'
local_load_dir = '/Volumes/External HDD/dataset/tacred/saved_models'
cwd = os.getcwd()
on_server = 'Desktop' not in cwd
base_load_dir = server_load_dir if on_server else local_load_dir
binary_dir_index = 'binary_model' if on_server else ''
positive_dir_index = 'positive_model' if on_server else ''

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='Directory of the model.',
                    default='/usr0/home/gis/research/tacred-exploration/saved_models/PA-LSTM-TACRED')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='/usr0/home/gis/data/tacred/data/json')
parser.add_argument('--vocab_dir', type=str,
                    default='/Volumes/External HDD/dataset/tacred/data/vocab')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--binary_model_file', type=str,
                    default=os.path.join(base_load_dir, 'PA-LSTM-TACRED-binary', binary_dir_index),
                    )
parser.add_argument('--positive_model_file', type=str,
                    default=os.path.join(base_load_dir, 'PA-LSTM-TACRED-binary', positive_dir_index)
                    )

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

# Load binary model
binary_model_file = os.path.join(args.binary_model_file, args.model)
print("Loading model from {}".format(binary_model_file))
binary_opt = torch_utils.load_config(binary_model_file)
binary_opt['apply_binary_classification'] = True
binary_model = RelationModel(binary_opt)
binary_model.load(binary_model_file)
# Load positive model
positive_model_file = os.path.join(args.positive_model_file, args.model)
print("Loading model from {}".format(positive_model_file))
positive_opt = torch_utils.load_config(positive_model_file)
positive_opt['apply_binary_classification'] = False
positive_opt['num_class'] = 42
positive_model = RelationModel(positive_opt)
positive_model.load(positive_model_file)

# load vocab
vocab_file = args.vocab_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
print('config vocab size: {} | actual size: {}'.format(
    binary_opt['vocab_size'], vocab.size
))
print('Binary config: {}'.format(binary_opt))
print('Positive config: {}'.format(positive_opt))
assert binary_opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
# TODO: May be necessary to save and load the mappings for each data iterator from the training.
#  I don't think this is neeeded atm because the data is loaded in exactly the same way between
#  the eval and train files --> same dictionary mapping values.
# load data
data_processor = DataProcessor(config=binary_opt,
                               vocab=vocab,
                               data_dir = binary_opt['data_dir'],
                               partition_names=['dev', 'test'])

dev_iterator =  data_processor.create_iterator(
config={
            'binary_classification': False,
            'exclude_negative_data': False,
            'relation_masking': False,
            'word_dropout': binary_opt['word_dropout']
        },
        partition_name='dev'
)
test_iterator =  data_processor.create_iterator(
config={
            'binary_classification': False,
            'exclude_negative_data': False,
            'relation_masking': False,
            'word_dropout': binary_opt['word_dropout']
        },
        partition_name='test'
)

helper.print_config(binary_opt)
helper.print_config(positive_opt)
binary_label2id = data_processor.name2id['binary_rel2id']
binary_id2label = dict([(v, k) for k, v in binary_label2id.items()])
label2id = data_processor.name2id['rel2id']
id2label = dict([(v,k) for k,v in label2id.items()])

def extract_eval_probs(dataset, model):
    data_probs = []
    for i, batch in enumerate(dataset):
        batch_probs, _ = model.predict(batch)
        data_probs += batch_probs
    return np.array(data_probs)

def evaluate_joint_models(dataset, binary_model, positive_model, id2label, binary_id2label, threshold):
    binary_probs = extract_eval_probs(dataset=dataset, model=binary_model)
    positive_probs = extract_eval_probs(dataset=dataset, model=positive_model)
    binary_preds = (binary_probs > threshold).astype(int)
    binary_labels = np.array([binary_id2label[p] for p in binary_preds])
    positive_preds = np.argmax(positive_probs, axis=-1)
    positive_labels = np.array([id2label[p] for p in positive_preds])
    test_labels = []
    for binary_label, positive_label in zip(binary_labels, positive_labels):
        if binary_label == 'no_relation':
            test_labels.append(binary_label)
        else:
            test_labels.append(positive_label)
    metrics = scorer.score(dataset.labels, test_labels)
    return metrics

threshold = 0.5096710324287415 # Fill this in
evaluate_joint_models(
    dataset=dev_iterator,
    binary_model=binary_model,
    positive_model=positive_model,
    id2label=id2label,
    binary_id2label=binary_id2label,
    threshold=threshold
)
evaluate_joint_models(
    dataset=test_iterator,
    binary_model=binary_model,
    positive_model=positive_model,
    id2label=id2label,
    binary_id2label=binary_id2label,
    threshold=threshold
)

# predictions = []
# all_probs = []
# data = []
# correct_predictions = []
# incorrect_predictions = []
# for i, b in enumerate(batch):
#     preds, probs, _ = model.predict(b)
#     predictions += preds
#     all_probs += probs
#
# predictions = [id2label[p] for p in predictions]
# p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

# is_correct = np.array(predictions) == np.array(batch.gold())
# is_incorrect = np.array(predictions) != np.array(batch.gold())
# correct_predictions = np.arange(len(predictions))[is_correct]
# incorrect_predictions = np.arange(len(predictions))[is_incorrect]
# with open(data_file, 'rb') as handle:
#     data = np.array(json.load(handle))
# correct_data = data[correct_predictions].tolist()
# incorrect_data = data[incorrect_predictions].tolist()

# save_dir = os.path.join(opt['test_save_dir'], opt['id'], 'correctness')
# os.makedirs(save_dir, exist_ok=True)
# with open(os.path.join(save_dir, 'correct_data.pkl'), 'wb') as handle:
#     pickle.dump(correct_data, handle)
# with open(os.path.join(save_dir, 'incorrect_data.pkl'), 'wb') as handle:
#     pickle.dump(incorrect_data, handle)


# # save probability scores
# if len(args.out) > 0:
#     helper.ensure_dir(os.path.dirname(args.out))
#     with open(args.out, 'wb') as outfile:
#         pickle.dump(all_probs, outfile)
#     print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

