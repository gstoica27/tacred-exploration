import os
import numpy as np
import json
from collections import defaultdict

def create_kg(data):
    question2obj2freq = defaultdict(lambda: defaultdict(lambda: 0))
    for d in data:
        subject = d['subj_type']
        relation = d['relation']
        object = d['obj_type']
        question = (subject, relation)
        question2obj2freq[question][object] += 1
    return question2obj2freq

def print_question_diffs(source_data, query_data):
    source_questions = set(source_data.keys())
    query_questions = set(query_data.keys())
    common_questions = source_questions.intersection(query_questions)
    print('='*80)
    print('Overlapping Questions')
    print('=' * 80)
    for question in common_questions:
        source_objs = set(source_data[question].keys())
        query_objs = set(query_data[question].keys())
        common_objs = source_objs.intersection(query_objs)
        missing_objs = query_objs - source_objs
        common_data = sum([query_data[question][obj] for obj in common_objs])
        missing_data = sum([query_data[question][obj] for obj in missing_objs])
        print('{} | Num Missing E2s: {} | Overlaps: {}'.format(question, missing_data, common_data))
    print('=' * 80)
    print('Questions Missing in Train')
    print('=' * 80)
    missing_questions = query_questions - source_questions
    for question in missing_questions:
        query_objs = set(query_data[question].keys())
        missing_data = sum([query_data[question][obj] for obj in query_objs])
        print('{} | Num Missing E2s: {} | Overlaps: {}'.format(question, missing_data, 0))


if __name__ == '__main__':
    # Set files
    cwd = '/Users/georgestoica/Desktop/Research/tacred-exploration/dataset/semeval/data/json'
    train_file = os.path.join(cwd, 'train_new.json')
    dev_file = os.path.join(cwd, 'dev.json')
    test_file = os.path.join(cwd, 'test.json')
    # Load data
    train_data = json.load(open(train_file, 'r'))
    dev_data = json.load(open(dev_file, 'r'))
    test_data = json.load(open(test_file, 'r'))
    # Create KG
    train_kg = create_kg(train_data)
    dev_kg = create_kg(dev_data)
    test_kg = create_kg(test_data)
    # Print Differences
    print_question_diffs(source_data=train_kg, query_data=test_kg)