from stanfordcorenlp import StanfordCoreNLP
import os
import json
import numpy as np
from tqdm import tqdm

def load_json(filepath, by_line=True):
    data = []
    with open(filepath, 'r') as handle:
        if by_line:
            for line in handle:
                data.append(
                    json.loads(line)
                )
        else:
            data = json.load(handle)
    return data

def align_parse_and_source(parsed_tokens, source):
    i = 0
    misalignment = 0
    source_tokens = source['sentence']
    subj_start, subj_end = source['subj_start'], source['subj_end']
    obj_start, obj_end = source['obj_start'], source['obj_start']
    try:
        while i < len(parsed_tokens):
            parsed_token = parsed_tokens[i]
            source_token = source_tokens[i - misalignment]
            if parsed_token == source_token:
                i += 1
            else:
                partial_token = ''
                offset = 0
                while partial_token != source_token:
                    partial_token += parsed_tokens[i + offset]
                    offset += 1
                misalignment += offset - 1
                if i < subj_start:
                    subj_start = subj_start + offset - 1
                    subj_end = subj_end + offset - 1
                if i < obj_start:
                    obj_start = obj_start + offset - 1
                    obj_end = obj_end + offset - 1
                if i >= subj_start and i <= subj_end:
                    subj_start = subj_start
                    subj_end = subj_end + offset - 1
                if i >= obj_start and i <= obj_end:
                    obj_start = obj_start
                    obj_end = obj_end + offset - 1
                i += offset

        source['subj_start'] = subj_start
        source['subj_end'] = subj_end
        source['obj_start'] = obj_start
        source['obj_end'] = obj_end
        source['sentence'] = parsed_tokens
        source['failed'] = False
    except:
        source['failed'] = True
    return source


def read_json(path):
    return json.load(open(path))


def extract_nlp_components(text_file, core_nlp):
    components = []
    data = read_json(text_file)

    for idx, instance in tqdm(enumerate(data)):
        tokens = instance['sentence']

        sentence = " ".join(tokens).strip()
        parsed_tokens = core_nlp.word_tokenize(sentence)

        aligned_instance = align_parse_and_source(parsed_tokens, instance)
        if aligned_instance['failed']:
            continue

        subject_start = aligned_instance['subj_start']
        subject_end = aligned_instance['subj_end']

        object_start = aligned_instance['obj_start']
        object_end = aligned_instance['obj_end']
        _, token_ners = zip(*core_nlp.ner(sentence))
        token_ners = list(token_ners)
        _, token_pos = zip(*core_nlp.pos_tag(sentence))
        token_pos = list(token_pos)
        token_deprel, token_head = extract_dependencies(sentence, core_nlp=core_nlp)

        # QUALITY CHECK
        assert (len(parsed_tokens) == len(token_ners))
        assert (len(parsed_tokens) == len(token_pos))
        assert (len(parsed_tokens) == len(token_deprel))
        assert (len(parsed_tokens) == len(token_head))
        assert (subject_start >= 0)
        assert (subject_end < len(parsed_tokens))
        assert (object_start >= 0)
        assert (object_end < len(parsed_tokens))

        sample = {
            'id': idx,
            'token': parsed_tokens,
            'subj_start': subject_start,
            'subj_end': subject_end,
            'obj_start': object_start,
            'obj_end': object_end,
            'subj_type': instance['subj_type'],
            'obj_type': instance['obj_type'],
            'stanford_pos': token_pos,
            'stanford_ner': token_ners,
            'stanford_deprel': token_deprel,
            'stanford_head': token_head,
            'relation': instance['relation']
        }

        components.append(sample)
    return components

def extract_dependencies(sentence, core_nlp):
    dependencies = core_nlp.dependency_parse(sentence)
    deprels = [''] * len(dependencies)
    heads = [0] * len(dependencies)
    for node in dependencies:
        deprel, head, idx = node
        deprels[idx-1] = deprel
        heads[idx-1] = head
    return deprels, heads

def get_deprel_and_head(stanza_tokens):
    deprel = []
    head = []
    for token in stanza_tokens:
        deprel.append(token.deprel)
        head.append(token.head)
    return deprel, head

def load_data(data_file):
    with open(data_file, 'r') as handle:
        return json.load(handle)

def compute_entity_ner(ners):
    entity_ner = None
    for candidate_ner in ners:
        if entity_ner is None:
            entity_ner = candidate_ner
        elif candidate_ner != 'O' and entity_ner == 'O':
            entity_ner = candidate_ner
    return entity_ner

def augment_data(data, sentences, nlp):
    new_data = []
    for sample in data:
        # sample id is zero indexed but sentences are 1 indexed
        sample_id = str(int(sample['id']) + 1)
        sentence = sentences[sample_id]
        tokens = sample['token']
        # sentence = ' '.join(tokens)
        token_ners = nlp.ner(sentence)
        # Add NER
        ners = []
        for token, ner in token_ners:
            if ner.isupper():
                ners.append(ner)
            else:
                ners.append('O')
        assert (len(ners) == len(tokens))
        sample['stanford_ner'] = ners
        ss, se = sample['subj_start'], sample['subj_end']
        os, oe = sample['obj_start'], sample['obj_end']

        subject_ners = np.unique(np.array(ners)[ss: se+1])
        object_ners = np.unique(np.array(ners)[os:oe+1])

        subject_ner = compute_entity_ner(subject_ners)
        object_ner = compute_entity_ner(object_ners)
        sample['subj_type'] = subject_ner
        sample['obj_type'] = object_ner
        new_data.append(sample)
    return new_data

if __name__ == '__main__':
    core_nlp = StanfordCoreNLP(r'/Users/georgestoica/Desktop/icloud_desktop/stanford-corenlp-4.0.0')

    data_dir = '/Volumes/External HDD/nyt_full'
    train_file = os.path.join(data_dir, 'train.json')
    test_file = os.path.join(data_dir, 'test.json')

    # train_metadata = parse_corpus(train_file)
    # test_metadata = parse_corpus(test_file)
    # print('Extracted!')
    # raw_semevel_dir = '/Users/georgestoica/Desktop/SemEval2010_task8_all_data'
    # train_file = os.path.join(raw_semevel_dir,
    #                           'SemEval2010_task8_training',
    #                           'TRAIN_FILE.TXT')
    # test_file = os.path.join(raw_semevel_dir,
    #                          'SemEval2010_task8_testing_keys',
    #                          'TEST_FILE_FULL.TXT')
    #
    train_data = extract_nlp_components(train_file, core_nlp=core_nlp)
    train_save_file = os.path.join(data_dir, 'train_sampled.json')
    json.dump(train_data, open(train_save_file, 'w'))
    test_data = extract_nlp_components(test_file, core_nlp=core_nlp)
    test_save_file = os.path.join(data_dir, 'test_new.json')
    json.dump(test_data, open(test_save_file, 'w'))

    core_nlp.close()