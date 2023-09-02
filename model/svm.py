from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json
from random import shuffle

import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


def preprocess(text):
    for c in ".,'" + '"?!()[]{}':
        text = text.replace(c, f" {c} ")
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip()
    return text


data = json.load(open("dictionary_translate/data/scores.json", "r", encoding="utf8"))
keys = list(data.keys())
# shuffle(keys)

num_train = int(0.9 * len(keys))

vi = [preprocess(item.lower()) for item in keys]
l = np.array([len(item.split()) for item in vi])
l = np.expand_dims(l, axis=1)
l = l / np.max(l) * 0.01
y = np.array([np.array(data[item]) for item in keys])
y = np.argmax(y, axis=1)

feature_extraction = TfidfVectorizer(analyzer="word", lowercase=False, norm="l2", ngram_range=(1, 1))
x = feature_extraction.fit_transform(vi)
x = x.toarray()
# x = np.concatenate([x, l], axis=1)

train_x = x[:num_train]
train_y = y[:num_train]
valid_x = x[num_train:]
valid_y = y[num_train:]

clf = SVC(probability=True, kernel='rbf')
clf.fit(train_x, train_y)

predictions = clf.predict_proba(valid_x)
prediction_label = np.argmax(predictions, axis=1)

print('ACC :' + str(np.sum(valid_y == prediction_label) / len(valid_y)))


