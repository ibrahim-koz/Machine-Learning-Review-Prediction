import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

lines = open("all_sentiment_shuffled.txt", "r", encoding="utf8")
x = []
y = []
first_column = []
third_column = []
for num, line in enumerate(lines):
    a = line.rstrip("\n").split(" ", 3)
    if a[1] == "neg":
        y.append(0)
    elif a[1] == "pos":
        y.append(1)
    x.append(a[0] + " " + a[2] + " " + a[3])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

pos_x_train = []
neg_x_train = []
for i, j in zip(x_train, y_train):
    if j == 0:
        neg_x_train.append(i)
    elif j == 1:
        pos_x_train.append(i)

posterior_pos_x = len(pos_x_train) / (len(pos_x_train) + len(neg_x_train))
posterior_neg_x = len(neg_x_train) / (len(pos_x_train) + len(neg_x_train))

count_vectorizer = CountVectorizer(ngram_range=(1, 2),  # to use bigrams ngram_range=(2,2)
                                   analyzer='word')

tfid_vectorizer = TfidfVectorizer(ngram_range=(1, 2),  # to use bigrams ngram_range=(2,2)
                                  analyzer='word')


def naive_bayes(l_count_vectorizer):
    pos_x_train_ar = l_count_vectorizer.fit_transform(pos_x_train).toarray()
    pos_indices = l_count_vectorizer.vocabulary_
    pos_freq = np.sum(pos_x_train_ar, axis=0)
    sum_pos_freq = np.sum(pos_freq)
    neg_x_train_ar = l_count_vectorizer.fit_transform(neg_x_train).toarray()
    neg_indices = l_count_vectorizer.vocabulary_
    neg_freq = np.sum(neg_x_train_ar, axis=0)
    sum_neq_freq = np.sum(neg_freq)
    alpha = 1
    predictions = []
    for r in x_test:
        row = l_count_vectorizer.fit_transform([r]).toarray()[0]
        pos_prob = math.log(posterior_pos_x)
        neg_prob = math.log(posterior_neg_x)
        for vocab, freq in zip(l_count_vectorizer.vocabulary_.keys(), row):
            if vocab in pos_indices:
                pos_prob += math.log((pos_freq[pos_indices[vocab]] + alpha) / (sum_pos_freq + alpha * len(pos_freq))) * freq
            else:
                pos_prob += math.log(alpha / (sum_pos_freq + alpha * len(pos_freq))) * freq

            if vocab in neg_indices:
                neg_prob += math.log((neg_freq[neg_indices[vocab]] + alpha) / (sum_neq_freq + alpha * len(neg_freq))) * freq
            else:
                neg_prob += math.log(alpha / (sum_neq_freq + alpha * len(neg_freq))) * freq

        if pos_prob > neg_prob:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


predictions = naive_bayes(count_vectorizer)
tfid_predictions = naive_bayes(tfid_vectorizer)

print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, tfid_predictions))
print(accuracy_score(y_test, tfid_predictions))
