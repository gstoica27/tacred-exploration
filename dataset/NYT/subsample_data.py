import os
import numpy as np
import json
from collections import defaultdict
import unicodedata
import stanza
from stanfordcorenlp import StanfordCoreNLP
from copy import deepcopy

def read_data(path):
    with open(path, 'r') as handle:
        sentences = handle.readlines()
    return sentences

def extract_sentence_level_data(data, max_tokens=100):
    # instances = []
    sentences = []
    id2triple = {}
    ids = []
    for raw_sentence in data:
        sentence_data = json.loads(raw_sentence.strip('\r\n'))
        # sentence_data = eval(raw_sentence)
        sentence_id = sentence_data['sentId']
        article_id = sentence_data['articleId']
        instance_id = f'{article_id}-{sentence_id}'
        sentence = sentence_data['sentText'].strip('\r\n').replace("''", "")
        sentence = unicodedata.normalize('NFKD',
                                         sentence). \
            encode('ascii', 'ignore')
        sentence_tokens = sentence.split()
        if len(sentence_tokens) > max_tokens:
            continue
        for idx, triple in enumerate(sentence_data['relationMentions']):
            subject = unicodedata.normalize('NFKD',
                                            triple['em1Text']).\
                encode('ascii','ignore')
            relation = triple['label']
            object = unicodedata.normalize('NFKD',
                                           triple['em2Text']). \
                encode('ascii', 'ignore')

            triple = (subject, relation, object)
            if instance_id not in id2triple:
                id2triple[instance_id] = []
            id2triple[instance_id].append(triple)

        sentences.append(sentence)
        ids.append(instance_id)
            # instance = {
            #     'sentence': sentence,
            #     'subject': subject,
            #     'relation': relation,
            #     'object': object,
            #     'id': instance_id + f'-{idx}'
            # }
            # instances.append(instance)
    instances = {
        'sentences': sentences,
        'id2triple': id2triple,
        'ids': ids
    }
    return instances

def subsample_sentences(sentence_data, keep_prop=.1):
    sample_amount = int(len(sentence_data) * keep_prop)
    return np.random.choice(sentence_data, sample_amount, replace=False)

def extract_parse_information(struct):
    info = {'token': [], 'stanford_deprel': [], 'stanford_head': [], 'stanford_pos': []}
    for word in struct.words:
        info['token'].append(word.text)
        info['stanford_deprel'].append(word.deprel)
        info['stanford_head'].append(word.head)
        info['stanford_pos'].append(word.xpos)
    return info

def match_ner_to_tokens(base_tokens, ners):
    matched_ners = []
    extracted_idx = 0
    for base_token in base_tokens:
        if extracted_idx >= len(ners):
            return None
            # raise ValueError('NERs cannot be matched. Base: {} \n| NERs: {}'.format(
            #     base_tokens, ners
            # ))
        compare_token, extracted_ner = ners[extracted_idx]
        ner = extracted_ner
        if base_token == compare_token:
            extracted_idx += 1
        else:
            spread_ners = set()
            while compare_token != base_token:
                spread_ners.add(extracted_ner)
                extracted_idx += 1
                if extracted_idx >= len(ners):
                    return None
                    # raise ValueError('NERs cannot be matched. Base: {} \n| NERs: {}'.format(
                    #     base_tokens, ners
                    # ))
                extracted_token, extracted_ner = ners[extracted_idx]
                compare_token += extracted_token
            extracted_idx += 1
            spread_ners.add(extracted_ner)
            ner = 'O'
            for spread_ner in spread_ners:
                if ner == 'O' and spread_ner != 'O':
                    ner = spread_ner
        matched_ners.append(ner)
    return matched_ners

def match_ner(base_tokens, ners):
    matched_ners = []
    extracted_idx = 0
    base_idx = 0
    base_compare = ''
    ner_compare = ''
    ner_set = set()
    tokens_to_be_set = 0
    base_compare_used = set()
    ner_compare_used = set()
    while base_idx < len(base_tokens) and extracted_idx < len(ners):
        ner_token, ner = ners[extracted_idx]
        base_token = base_tokens[base_idx]
        # Don't double count tokens in same parse
        if base_idx not in base_compare_used:
            base_compare += base_token
            base_compare_used.add(base_idx)
        if extracted_idx not in ner_compare_used:
            ner_compare += ner_token
            ner_compare_used.add(extracted_idx)

        # Case #1: NER Token and Base Token are same
        if base_compare == ner_compare:
            tokens_to_be_set += 1
            ner_set.add(ner)
            # Choose NER to match to base tokens
            chosen_ner = 'O'
            for ner in ner_set:
                if ner != 'O' and chosen_ner == 'O':
                    chosen_ner = ner
            # Spread NER over number of tokens needed
            base_ner = [chosen_ner] * tokens_to_be_set
            matched_ners += base_ner
            # Move indices to next step
            base_idx += 1
            extracted_idx += 1
            tokens_to_be_set = 0
            # Reset match components
            base_compare = ''
            ner_compare = ''
            ner_set = set()
            ner_compare_used = set()
            base_compare_used = set()
        elif len(base_compare) < len(ner_compare):
            tokens_to_be_set += 1
            base_idx += 1
        elif len(base_compare) > len(ner_compare):
            extracted_idx += 1
            ner_set.add(ner)
        else:
            return matched_ners
    return matched_ners

def get_entity_type(ners, span):
    ner_span = ners[span[0]: span[1] + 1]
    entity_ner = 'O'
    for ner in ner_span:
        if ner != 'O':
            entity_ner = ner
            return entity_ner
    return entity_ner

def get_triple_parse(tokens, triples, parse_dict):
    entity2span = {}
    entity2type = {}

    all_parsed = []
    for triple in triples:
        subject, relation, object = triple
        if subject not in entity2span:
            subject_span = find_entity_span(tokens, subject)
            if subject_span is None: continue
            entity2span[subject] = subject_span
        if object not in entity2span:
            object_span = find_entity_span(tokens, object)
            if object_span is None: continue
            entity2span[object] = object_span
        complete_parse = deepcopy(parse_dict)
        subject_span = entity2span[subject]
        object_span = entity2span[object]

        if subject not in entity2type:
            subject_type = get_entity_type(complete_parse['stanford_ner'], subject_span)
            entity2type[subject] = subject_type
        if object not in entity2type:
            object_type = get_entity_type(complete_parse['stanford_ner'], object_span)
            entity2type[object] = object_type

        complete_parse['subj_start'], complete_parse['subj_end'] = subject_span
        complete_parse['obj_start'], complete_parse['obj_end'] = object_span
        complete_parse['subj_type'] = entity2type[subject]
        complete_parse['obj_type'] = entity2type[object]
        complete_parse['relation'] = relation
        all_parsed.append(complete_parse)
    return all_parsed


def find_entity_span(tokens, entity):
    span_start = None
    entity_tokens = entity.split()
    entity_idx = 0
    matched = False
    for idx, token in enumerate(tokens):
        if entity_tokens[entity_idx].decode('ascii') == token:
            matched = True
            if span_start is None:
                span_start = idx
            entity_idx += 1
        elif token != entity_tokens[entity_idx]:
            span_start = None
            matched = False
            entity_idx = 0
        if entity_idx >= len(entity_tokens) and matched:
            return (span_start, idx)
    return None


def parse_sentences(sentence_data, batch_size=100):
    sentences, sentence_ids, id2triple = sentence_data['sentences'], sentence_data['ids'], sentence_data['id2triple']

    parsed_data = []
    total_parsed_error = 0
    total_ner_error = 0
    total_id_error = 0
    stanza.download('en')
    core_nlp = StanfordCoreNLP(r'/Users/georgestoica/Desktop/stanford-corenlp-4.0.0')
    # core_nlp = StanfordCoreNLP(os.path.join(os.getcwd(), 'stanford-corenlp-4.0.0'))
    nlp = stanza.Pipeline('en', use_gpu=True)
    for batch_start in range(0, len(sentences), batch_size):
        batch_end = batch_start + batch_size
        # Create batches
        batch_sentences = sentences[batch_start:batch_end]
        batch_ids = sentence_ids[batch_start:batch_end]
        # Batch sentences for NLP parse
        combined_sentences = ''
        for batch_sentence in batch_sentences:
            combined_sentences += batch_sentence.decode('ascii').strip() + '\n\n'
        # Obtain POS, DEPREL, TOKEN, HEAD
        parsed_struct = nlp(combined_sentences).sentences
        if len(parsed_struct) > 1:
            total_parsed_error += 1
            continue
        for batch_idx in range(0, len(batch_sentences)):
            sentence_struct = parsed_struct[batch_idx]
            sentence_id = batch_ids[batch_idx]
            # Obtain NER
            parsed_instance = extract_parse_information(sentence_struct)
            unmatched_ners = core_nlp.ner(batch_sentences[batch_idx].decode('ascii'))
            matched_ners = match_ner(base_tokens=parsed_instance['token'],
                                               ners=unmatched_ners)
            if len(matched_ners) != len(parsed_instance['token']):
                total_ner_error += 1
                continue
            parsed_instance['stanford_ner'] = matched_ners
            if sentence_id not in id2triple:
                total_id_error += 1
                continue
            parsed_sentence_triples = get_triple_parse(tokens=parsed_instance['token'],
                                                       triples=id2triple[sentence_id],
                                                       parse_dict=parsed_instance)
            parsed_data += parsed_sentence_triples
    print('parsed error: {} | ner eror: {} | id error: {}'.format(
        total_parsed_error, total_ner_error, total_id_error))
    return parsed_data


if __name__ == '__main__':
    partition_name = 'dev'
    cwd = os.getcwd()
    partition_file = os.path.join(cwd, f'{partition_name}.json')
    # single_sentence = os.path.join(cwd, 'single_example.txt')
    # single_data = read_data(single_sentence)
    # single_sentences = extract_sentence_level_data(single_data)
    # single_parsed = parse_sentences(single_sentences)
    data = read_data(partition_file)
    components = extract_sentence_level_data(data)
    parsed_data = parse_sentences(components, batch_size=1)

    save_file = os.path.join(cwd, f'{partition_name}_parsed.json')
    json.dump(parsed_data, open(save_file, 'w'))



