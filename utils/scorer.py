#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import numpy as np
from collections import defaultdict

NO_RELATION = "no_relation"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args

def score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()
    positive_guesses = Counter()


    actual_fn = 0
    actual_fp = 0
    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
         
        # if gold == NO_RELATION and guess == NO_RELATION:
        if NO_RELATION in gold and NO_RELATION in guess:
            pass
        # elif gold == NO_RELATION and guess != NO_RELATION:
        elif NO_RELATION in gold and NO_RELATION not in guess:
            guessed_by_relation[guess] += 1
            actual_fp += 1
        # elif gold != NO_RELATION and guess == NO_RELATION:
        elif NO_RELATION not in gold and NO_RELATION in guess:
            gold_by_relation[gold] += 1
            actual_fn += 1
        # elif gold != NO_RELATION and guess != NO_RELATION:
        elif NO_RELATION not in gold and NO_RELATION not in guess:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            positive_guesses[guess] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    TP = float(sum(correct_by_relation.values()))
    FP = float(sum(guessed_by_relation.values())) - float(sum(correct_by_relation.values()))
    FN = float(sum(gold_by_relation.values())) - float(sum(correct_by_relation.values()))
    total_positive_guessed = float(sum(positive_guesses.values()))
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    positive_accuracy = TP / max(1, total_positive_guessed)
    print( "Precision (micro): {:.3%}".format(prec_micro) )
    print( "   Recall (micro): {:.3%}".format(recall_micro) )
    print( "       F1 (micro): {:.3%}".format(f1_micro) )
    print("               TP: {:.3f}".format(TP))
    print("               FP: {:.3f}".format(FP))
    print("               FN: {:.3f}".format(FN))
    print("Positive Accuracy: {:.3%}".format(positive_accuracy))
    metrics = {'precision': prec_micro,
               'recall': recall_micro,
               'f1': f1_micro,
               'TP': TP, 'FP': FP,
               'FN': FN,
               'pos_acc': positive_accuracy}
    return metrics

def compute_confusion_matrices(ground_truth, predictions):
    confusion_matrix = {}
    for correct, prediction in zip(ground_truth, predictions):
        if correct not in confusion_matrix:
            confusion_matrix[correct] = {}
        if prediction not in confusion_matrix[correct]:
            confusion_matrix[correct][prediction] = 0
        confusion_matrix[correct][prediction] += 1
    return confusion_matrix


if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, verbose=True)

