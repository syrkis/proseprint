# utils.py
#   utility functions for the project
# by: Noah Syrkis

# imports
import os
import pandas as pd
from tqdm import tqdm
import json
import argparse
from sklearn.metrics import f1_score
from collections import Counter, defaultdict



# constants
data_path = 'data/release'
PAD = '<PAD>'
UNK = '<UNK>'
PAD_IDX = 0
UNK_IDX = 1
dataset_names = [f'pan23-multi-author-analysis-dataset{i}' for i in range(1, 4)]


encode = lambda x, stoi: [stoi.get(w, UNK_IDX) for w in x]
decode = lambda x, itos: ''.join([itos.get(i, UNK) for i in x])
flatten = lambda x: [i for j in x for i in j]


# functions
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', help='model to use')
    parser.add_argument('--dataset', type=str, default='1', help='dataset to use')
    parser.add_argument('--process', type=str, default='siam', help='preprocessing to use')
    args = parser.parse_args()
    args.dataset = dataset_names[int(args.dataset) - 1]
    return args


def evaluate_predictions(name, y_true, y_pred):
    print(name)
    print('Weigthed F1:', f1_score(y_true, y_pred, average='weighted'))
    print()



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


def get_paired_dataset(dataset):
    D = []
    for i in range(len(dataset)):
        for j in range(len(dataset.iloc[i]['changes'])):
            doc1 = dataset.iloc[i]['text'][j]
            doc2 = dataset.iloc[i]['text'][j+1]
            change = dataset.iloc[i]['changes'][j]
            D.append({'doc1': doc1, 'doc2': doc2, 'change': change}) 
    df = pd.DataFrame(D, columns=['doc1', 'doc2', 'change'])
    return df


def make_word_vocab(dataset, min_count=50):
    # make a vocabulary from the training set
    vocab = Counter(flatten(flatten(dataset['text'])))
    vocab = [w for w, c in vocab.most_common() if c > min_count] + ['\n', ' '] + list('abcdefghijklmnopqrstuvwxyz') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    vocab = [PAD, UNK] + sorted(list(set(vocab)))
    # return UNK if word is not in vocab
    stoi = defaultdict(lambda: 1, {w: i for i, w in enumerate(vocab)})
    itos = {i: w for i, w in enumerate(vocab)}
    return stoi, itos


def make_char_vocab(dataset, min_count=50):
    # make a vocabulary from the training set
    vocab = Counter(flatten(flatten(dataset['text'])))
    vocab = [w for w, c in vocab.most_common() if c > min_count] + ['\n', ' '] + list('abcdefghijklmnopqrstuvwxyz') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    vocab = [PAD, UNK] + sorted(list(set(vocab)))
    # return UNK if word is not in vocab
    stoi = defaultdict(lambda: 1, {w: i for i, w in enumerate(vocab)})
    itos = {i: w for i, w in enumerate(vocab)}
    return stoi, itos