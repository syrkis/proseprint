# utils.py
#   utility functions for the project
# by: Noah Syrkis

# imports
import os
import pandas as pd
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer
import argparse
from sklearn.metrics import f1_score
from collections import Counter, defaultdict


# constants
DATA_PATH = os.path.join(os.getcwd(), 'data/release/')
DATASET_NAMES = [f'pan23-multi-author-analysis-dataset{i}' for i in range(1, 4)]

# functions
def get_data(dataset_id, transform_fn):
    dataset = DATASET_NAMES[dataset_id - 1]
    data = {'train': None, 'validation': None}
    for split in data.keys():
        split_dir = os.path.join(DATA_PATH, dataset, dataset + '-' + split)
        split_data = get_split_data(split_dir, transform_fn)
        data[split] = split_data
    return data


def get_split_data(split_dir, transform_fn):
    data = {}
    model = SentenceTransformer('all-MiniLM-L6-v2')

    truth_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
    for f in truth_files:
        problem_id = f.split('.')[0].split('-')[-1]
        with open(os.path.join(split_dir, f), 'r') as f:
            data[problem_id] = {'truth': json.load(f), 'text': None }

    text_files = [f for f in os.listdir(split_dir) if f.endswith('.txt')]
    for f in tqdm(text_files):
        problem_id = f.split('.')[0].split('-')[-1]
        with open(os.path.join(split_dir, f), 'r') as f:
            lines = f.readlines()
            data[problem_id]['text'] = model.encode(lines)
        
    return data