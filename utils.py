# utils.py
#   utility functions for the project
# by: Noah Syrkis

# imports
import os
import pandas as pd
from tqdm import tqdm
import json


# constants
data_path = 'data/release'


# functions
def get_files(dataset, split='train', data_path=data_path):
    folder_path = os.path.join(data_path, dataset, dataset + '-' + split)
    files = os.listdir(folder_path)
    files = make_pairs(files, folder_path)
    return files


def make_pairs(files, folder_path):
    # there are two files for each sample problem-id.txt and truth-problem-id.json
    # we want to pair them up
    pairs = []
    for f in files:
        if f.endswith('.txt'):
            truth_file = 'truth-' + f.replace('.txt', '.json')
            pairs.append((os.path.join(folder_path, f), os.path.join(folder_path, truth_file)))
    return pairs


def make_dataset(files):
    D = []
    for text_file, truth_file in tqdm(files):
        with open(text_file, 'r') as f:
            text = f.read()
        with open(truth_file, 'r') as f:
            truth = json.load(f)
        D.append({'id': text_file.split('/')[-1][8:-4], 'text': text.split('\n'), 'authors': truth['authors'], 'changes': truth['changes']})
    df = pd.DataFrame(D, index=[d['id'] for d in D], columns=['text', 'authors', 'changes'])
    return df


def get_data(dataset, split='train', data_path=data_path):
    files = get_files(dataset, split, data_path)
    df = make_dataset(files)
    return df