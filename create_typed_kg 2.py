import numpy as np
import os
import json
from utils import constant


data_dir = '/Volumes/External HDD/dataset/tacred/data/json'
filenames = ['train', 'test', 'dev']
write_dir = '/Users/georgestoica/Desktop/Research/ConvE/data/TACRED'

def create_partition_kg(data_dir, filen_name, write_dir, write_name):
    data_file = os.path.join(data_dir, filen_name)
    write_file = os.path.join(write_dir , write_name)

    with open(data_file) as handle:
        data = json.load(handle)
    with open(write_file, 'w') as handle:
        for sample in data:
            subj_type = sample['subj_type']
            obj_type = sample['obj_type']
            relation = sample['relation']

            # line = f'SUBJ-{subj_type}\t{relation}\tOBJ-{obj_type}\n'
            # line = f'SUBJ-{subj_type}\tOBJ-{obj_type}\t{relation}\n'
            line = f'{subj_type}\t{relation}\t{obj_type}\n'
            handle.write(line)

create_partition_kg(data_dir, 'train.json', write_dir, 'train.txt')
create_partition_kg(data_dir, 'test.json', write_dir, 'test.txt')
create_partition_kg(data_dir, 'dev.json', write_dir, 'valid.txt')