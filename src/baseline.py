# baseline.py
#     baseline model for mawsa
# by: Noah Syrkis

# imports
from src.utils import get_paired_dataset, PAD, UNK, evaluate_predictions
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import numpy as np


# baseline
def run_baseline(train_dataset, valid_dataset):
    base_one_train_data = get_paired_dataset(train_dataset)
    base_one_valid_data = get_paired_dataset(valid_dataset)
    chars = [PAD, UNK] + list('abcdefghijklmnopqrstuvwxyz') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['\n', ' ', '.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '-', '_', '/', '\\', '|', '[', ']', '{', '}', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '`', '~']
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for i, c in enumerate(chars)}
    encode = lambda x: [ctoi.get(c, 1) for c in x]
    decode = lambda x: "".join([itoc[i] for i in x])
    base_one_train_data[['doc1_idx', 'doc2_idx']] = base_one_train_data[['doc1', 'doc2']].applymap(encode)
    base_one_valid_data[['doc1_idx', 'doc2_idx']] = base_one_valid_data[['doc1', 'doc2']].applymap(encode)

    tfidf = TfidfVectorizer(max_features=200)
    tfidf.fit(base_one_train_data['doc1'] + base_one_train_data['doc2'])
    base_one_train_data['doc1_tfidf'] = tfidf.transform(base_one_train_data['doc1']).toarray().tolist()
    base_one_train_data['doc2_tfidf'] = tfidf.transform(base_one_train_data['doc2']).toarray().tolist()
    base_one_valid_data['doc1_tfidf'] = tfidf.transform(base_one_valid_data['doc1']).toarray().tolist()
    base_one_valid_data['doc2_tfidf'] = tfidf.transform(base_one_valid_data['doc2']).toarray().tolist()

    X_train = np.array(base_one_train_data['doc1_tfidf'].tolist()) - np.array(base_one_train_data['doc2_tfidf'].tolist())
    y_train = base_one_train_data['change']
    X_valid = np.array(base_one_valid_data['doc1_tfidf'].tolist()) - np.array(base_one_valid_data['doc2_tfidf'].tolist())
    y_valid = base_one_valid_data['change']


    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        reg_alpha=0.1,
        reg_lambda=10,
        )
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)


    evaluate_predictions('Train XGBoost', y_train, pred_train)
    evaluate_predictions('Valid XGBoost', y_valid, pred_valid)
    evaluate_predictions('Train ones', y_train, np.ones_like(y_train))
    evaluate_predictions('Valid ones', y_valid, np.ones_like(y_valid))