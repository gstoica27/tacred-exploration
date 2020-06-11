import os
import json

def get_max_tokens(data, start_value=0):
    max_tokens = start_value
    for d in data:
        num_tokens = len(d['token'])
        if num_tokens > max_tokens:
            max_tokens = num_tokens
    return max_tokens

if __name__ == '__main__':
    cwd = os.getcwd()
    files = ['train_split_parsed.json', 'dev_parsed.json', 'test_parsed.json']
    max_tokens = 0
    for filename in files:
        path = os.path.join(cwd, filename)
        data = json.load(open(path, 'r'))
        max_tokens = get_max_tokens(data, start_value=max_tokens)
    print(max_tokens)
