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
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import numpy as np


# constants
DATA_PATH = os.path.join(os.getcwd(), 'data/release/')
DATASET_NAMES = [f'pan23-multi-author-analysis-dataset{i}' for i in range(1, 4)]

# functions
def get_data(dataset_id):
    dataset = DATASET_NAMES[dataset_id - 1]
    data = {'train': None, 'validation': None}
    for split in data.keys():
        split_dir = os.path.join(DATA_PATH, dataset, dataset + '-' + split)
        split_data = get_split_data(split_dir)
        data[split] = split_data
    # rename validation to valid
    valid_idx, test_idx = train_test_split(list(data['validation'].keys()), test_size=0.5, random_state=42) 
    data['valid'] = {k: data['validation'][k] for k in valid_idx}
    data['test'] = {k: data['validation'][k] for k in test_idx}
    del data['validation']
    return data

def make_syntactic_features(paragraphs):
    features = []
    for paragraph in paragraphs:
        features.append(make_syntactic_features_paragraph(paragraph))
    features = np.array(features)
    return features

def make_syntactic_features_paragraph(paragraph):
    # construct a feature vector for a paragraph with syntactic features and other information
    feature_vector = np.zeros(61)
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph)
    pos_tags = pos_tag(words)
    pos_tags = [tag[1] for tag in pos_tags]
    pos_tag_counts = Counter(pos_tags)
    feature_vector[0] += len(sentences)
    feature_vector[1] += len(words)
    feature_vector[2] += len(set(words))
    feature_vector[3] += len(set(words)) / len(words)
    feature_vector[4] += len(pos_tag_counts)
    feature_vector[5] += len(pos_tag_counts) / len(words)
    feature_vector[6] += len(pos_tag_counts) / len(sentences)
    tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'PRP', 'PRP$']
    for idx, tag in enumerate(tags):
        feature_vector[(idx * 3) + 7] += pos_tag_counts[tag]
        feature_vector[(idx * 3) + 8] += pos_tag_counts[tag] / len(words)
        feature_vector[(idx * 3) + 9] += pos_tag_counts[tag] / len(sentences)
    return feature_vector


def get_split_data(split_dir):
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
            data[problem_id]['text'] = lines
            data[problem_id]['semantic'] = model.encode(lines)
            data[problem_id]['syntactic'] = make_syntactic_features(lines)
        
    return data