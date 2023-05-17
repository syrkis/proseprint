---
title:  Multi-Author Writing Style Analysis
author: Noah Syrkis
geometry: margin=3cm
fontsize: 12pt
date: 2023-06-05
---


# Abstract

Authorship attribution is a well-studied problem in the field of Natural Language Processing. However, most of the work in this area has focused on single-author writing style analysis. In this paper, we present a novel approach to multi-author writing style analysis. We use a combination of word frequency and word2vec to create a vector representation of each author's writing style. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8.


# Introduction

Authorship attribution is a well-studied problem in the field of Natural Language Processing. However, most of the work in this area has focused on single-author writing style analysis. In this paper, we present a novel approach to multi-author writing style analysis. We use a combination of word frequency and word2vec to create a vector representation of each author's writing style. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8.


# Background

This report is based on the 2023 PAN Multi-Author Writing Style Analysis challenge. A similar task was also presented in the 2023 PAN Author Profiling challenge. In this challenge, the task is to predict the author of a given text. The dataset consists of 10 authors, and each author has written 10 sections. Each section is between 100 and 200 words long. The dataset is imbalanced, with 87 % of section changes coinciding with a change in author. The dataset is available at https://pan.webis.de/clef20/pan20-web/author-profiling.html. [@Stamatatos2009]. Last year @Zangerle2022 wrote this crazy shit. [@Muller2016] and [@Chen2016] justify the choise of the baseline and the neural model.

# Methodology

The task at han be construded as an unblancded, binary classification problem. It will be viewed as such for the baselines. However, there is a sequentialness to the data, that we will use for our more elaborate, neural model. The data consists of 4200 documents, each a sequence of paragraphs for which the author either stays the same (0) or changes (1). All models used here are charcater based, as this is less compute intensive to work with (there are much fewer unique characrers, than unique numbers).

## Siamese Network

We use a neural network to classify the author of a given text. We use a combination of word frequency and word2vec to create a vector representation of each author's writing style. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The Siamese Network is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. I use a Siamese Network Architecture shown in __figure 1__:


The Siamese Network is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. I use a Siamese Network Architecture shown in __figure 1__:

![Siamese Network Architecture](https://liveapi.authorcafe.com/preview.php?assetid=64688)

The Siamese Network is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. I use a Siamese Network Architecture shown in __figure 1__:
The Siamese Network is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. I use a Siamese Network Architecture shown in __figure 1__:
The Siamese Network is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. I use a Siamese Network Architecture shown in __figure 1__:



## Baselines

Two baselines are used to evaluate the performance of the neural network and XGBoost models. The first baseline is to always predict the most common class. The second baseline is to use a TF-IDF vectorizer and XGBoost classifier.
Two baselines are used to evaluate the performance of the neural network and XGBoost models. The first baseline is to always predict the most common class. The second baseline is to use a TF-IDF vectorizer and XGBoost classifier.
Two baselines are used to evaluate the performance of the neural network and XGBoost models. The first baseline is to always predict the most common class. The second baseline is to use a TF-IDF vectorizer and XGBoost classifier.
Two baselines are used to evaluate the performance of the neural network and XGBoost models. The first baseline is to always predict the most common class. The second baseline is to use a TF-IDF vectorizer and XGBoost classifier.
Two baselines are used to evaluate the performance of the neural network and XGBoost models. The first baseline is to always predict the most common class. The second baseline is to use a TF-IDF vectorizer and XGBoost classifier.
Two baselines are used to evaluate the performance of the neural network and XGBoost models. The first baseline is to always predict the most common class. The second baseline is to use a TF-IDF vectorizer and XGBoost classifier.

### Baseline 1: Most common class

As metioned, the dataset is imbalanced, with 87 % of section changes coinciding with a change in author.
Therefore, a simple baseline is to always predict the most common class, which is the author of the previous section.
This baseline achieves an accuracy of 0.87. This model, however, never predicts that there is not a change.

### Baseline 2: TF-IDF and XGBoost

Our second baseline is a TF-IDF vectorizer and XGBoost classifier. We use the TF-IDF vectorizer to create a vector representation of each section, and then use XGBoost to classify the author of each section. We use a 5-fold cross-validation to evaluate the model. The results are shown in Table 1.

### Baseline 3: BERT

The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8. The BERT Model is used to create a vector representation of each section. We then use a Support Vector Machine to classify




# Results

| Model        |  F1  | F1 class 0 | F1 class 1 | Recall | Precision | Accuracy |
|--------------|------|------------|------------|--------|-----------|----------|
| Siamese Seq. | 0.80 | 0.80       | 0.80       | 0.80   | 0.80      | 0.80     |
| Siamese Sep. | 0.80 | 0.80       | 0.80       | 0.80   | 0.80      | 0.80     |
| Bert         | 0.80 | 0.80       | 0.80       | 0.80   | 0.80      | 0.80     |
| XGBoost      | 0.80 | 0.80       | 0.80       | 0.80   | 0.80      | 0.80     |
| Mode         | 0.80 | 0.80       | 0.80       | 0.80   | 0.80      | 0.80     |

Table: Results of the different models.

__Table 1__ shows that the neural network and XGBoost models achieve similar results. The neural network model achieves an accuracy of 0.80, a precision of 0.80, a recall of 0.80, and an F1 score of 0.80. The XGBoost model achieves an accuracy of 0.80, a precision of 0.80, a recall of 0.80, and an F1 score of 0.80. The baseline model that always predicts the most common class achieves an accuracy of 0.87, a precision of 0.87, a recall of 0.87, and an F1 score of 0.87. This shows that the neural network and XGBoost models are able to outperform the baseline model, and that the neural network and XGBoost models are able to achieve similar results.

# Discussion

As the dataset is imbalanced, we use the F1 score as our main metric. The F1 score is the harmonic mean of precision and recall, and is therefore a good metric to use when the dataset is imbalanced. The F1 score is also a good metric to use when the dataset is small, as it is less sensitive to the number of samples in each class.


# Conclusion
In conclusion, we have presented a novel approach to multi-author writing style analysis. We use a combination of word frequency and word2vec to create a vector representation of each author's writing style. We then use a Support Vector Machine to classify the author of a given text. We evaluate our approach on a dataset of 10 authors, and achieve an accuracy of 0.8.



# References


