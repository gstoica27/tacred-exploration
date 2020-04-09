"""
Helper functions.
"""

import os
import json
import argparse
from copy import deepcopy
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

### IO
def check_dir(d):
    if not os.path.exists(d):
        print("Directory {} does not exist. Exit.".format(d))
        exit(1)

def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            print("File {} does not exist. Exit.".format(f))
            exit(1)

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

def load_config(path, verbose=True):
    with open(path) as f:
        config = json.load(f)
    if verbose:
        print("Config loaded from file {}".format(path))
    return config

def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")
    return

class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)
    
    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)

def transform_labels(labels, no_rel_id=0):
    # Transform multi-label elements to binary label, by mapping
    # many -> 1 and 1 -> 1
    binary_labels = deepcopy(labels)
    binary_labels[labels != no_rel_id] = 1
    binary_labels[labels == no_rel_id] = 0
    return binary_labels

def filter_array(probs, filter_id):
    # Remove filter id column from array
    filtered_probs = deepcopy(probs)
    filter_mask = np.ones(probs.shape, dtype=np.bool)
    filter_mask[:, filter_id] = False
    return filtered_probs[:, filter_mask]

def choose_labels(probs, threshold, exclude_id):
    reduced_probs = filter_array(probs, filter_id=exclude_id)
    best_probs = np.argmax(reduced_probs, axis=1)

def create_predictions(probs, threshold, other_label=41):
    predictions = np.zeros(len(probs), dtype=np.int)
    best_probs = np.max(probs, axis=1)
    best_idxs = np.argmax(probs, axis=1)
    positive_pred = best_probs > threshold
    negative_pred = best_probs <= threshold
    predictions[negative_pred] = other_label
    predictions[positive_pred] = best_idxs[positive_pred]
    return predictions

def find_accuracy_threshold(probs, true_labels):
    """Probs: [N, L-1], True Labels: [N]"""
    reduced_probs = np.max(probs, axis=1)
    fpr, tpr, thresholds = roc_curve(true_labels, reduced_probs)

    num_pos = np.sum(true_labels == 1)
    num_neg = np.sum(true_labels == 0)
    tp = tpr * num_pos
    tn = (1 - fpr) * num_neg
    acc = (tp + tn) / (num_pos + num_neg)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold

def compute_one_vs_many_predictions(probs, true_label_names, rel2id, threshold=None):
    # Compute threshold to apply
    probs = np.array(probs)
    if threshold is None:
        true_labels = np.array([rel2id[label] for label in true_label_names])
        binary_labels = transform_labels(labels=true_labels, no_rel_id=rel2id['no_relation'])
        _, threshold = find_accuracy_threshold(probs=probs, true_labels=binary_labels)

    predictions = create_predictions(probs, threshold, other_label=rel2id['no_relation'])
    return predictions, threshold