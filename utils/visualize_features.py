import matplotlib.pyplot as plt
import numpy as np
import os



def plot_histogram(data, are_wrong, pair2rels, vocab):
    num_rels2correctness = {}
    for (d, is_wrong) in zip(data, are_wrong):
        subj_type = 'SUBJ-' + d['subj_type']
        obj_type = 'OBJ-' + d['obj_type']
        subject, object = vocab.word2id[subj_type], vocab.word2id[obj_type] - 4
        num_rels = len(pair2rels[(subject, object)])
        if num_rels not in num_rels2correctness.keys():
            num_rels2correctness[num_rels] = {'correct': 0, 'incorrect': 0}
        num_rels2correctness[num_rels]['correct'] += is_wrong != True
        num_rels2correctness[num_rels]['incorrect'] += is_wrong == True
    for num_rels in num_rels2correctness:
        correctness = num_rels2correctness[num_rels]
        print('Number relations: {} | Correct: {} | Incorrect: {}'.format(num_rels,
                                                                          correctness['correct'],
                                                                          correctness['incorrect']))
        correctness = num_rels2correctness[num_rels]
        accuracy = correctness['correct'] / (correctness['correct'] + correctness['incorrect'])
        num_rels2correctness[num_rels] = accuracy

    num_rels, accuracy = zip(*sorted(num_rels2correctness.items(),key=lambda item: item[0]))
    plt.plot(num_rels, accuracy, 'r-')
    plt.title('Accuracy by number of relations')
    plt.xlabel('Number of Relations')
    plt.ylabel('Accuracy')
    plt.show()


