import numpy as np
import os
import json
import sys

samples = []
with open(sys.argv[1], "r") as f:
    for sample in f:
        samples.append(sample)

stop_words = []

import re


def clean_data(sentense):
    x = sentense.split(' ')
    label1 = x[1]
    label2 = x[2]
    words = x[3:]
    output = []
    for word in words:
        word = word.lower()
        if word not in stop_words:
            output.append(word)
    output = " ".join(output)
    output = re.sub('[^a-zA-Z0-9\n\-]', ' ', output)
    output = re.sub('\s+', ' ', output)
    if label1 == "Fake":
        label1 = 0
    else:
        label1 = 1
    if label2 == "Pos":
        label2 = 1
    else:
        label2 = 0
    return [label1, label2], output


clean_sentense = []
target = []

for x in samples:
    a, b = clean_data(x)
    clean_sentense.append(b)
    target.append(a)

dic = {}
count = 0
word2index = {}
words = set()
for sentense in clean_sentense:
    for word in sentense.split(" "):
        if len(word) <= 1:
            continue
        else:
            if word not in dic:
                dic[word] = 1
                words.add(word)
                word2index[word] = count
                count += 1
            else:
                dic[word] += 1

features = np.zeros((len(clean_sentense), count))

for i, sentense in enumerate(clean_sentense):
    for word in sentense.split(" "):
        if word in words:
            features[i, word2index[word]] += 1
target = np.array(target)

print("123123123")


class Perception:
    def __init__(self, lr=1, n_iter=50):

        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        self.w = np.random.random((n_feature, 2))
        self.w = np.zeros((n_feature, 2))
        self.b = np.zeros((2))
        for _ in range(self.n_iter):
            for i, x in enumerate(X):
                x = x.reshape(1, x.shape[0])
                x_1 = np.dot(x, self.w) + self.b
                y_pred = self.activation_func(x_1)
                f_ = self.lr * (y[i] - y_pred)
                self.w += (x.T * f_)
                self.b += f_[0]

    def predict(self, x):
        x = np.dot(x, self.w)
        x[:, 0] += self.b[0]
        x[:, 1] += self.b[1]
        y_pred = self.activation_func(x)
        return y_pred

    def activation_func(self, x):
        return np.where(x > 0, 1, 0)


model1 = Perception()
model1.fit(features, target)
w = np.array(model1.w).tolist()
b = np.array(model1.b).tolist()
params = {
    "w": w,
    "b": b,
    "dic": dic,
    "count": count,
    "word2index": word2index,
    "words": list(words)
}
with open("vanillamodel.txt", "w") as f:
    json.dump(params, f)

print("123123123")


class Perception2:
    def __init__(self, lr=1, n_iter=30):

        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.u = 0
        self.beta = 0

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        self.w = np.random.random((n_feature, 2))
        self.w = np.zeros((n_feature, 2))
        self.b = np.zeros((2))
        cnt = 0

        for _ in range(self.n_iter):
            for i, x in enumerate(X):
                x = x.reshape(1, x.shape[0])
                x_1 = np.dot(x, self.w) + self.b
                y_pred = self.activation_func(x_1)
                f_ = self.lr * (y[i] - y_pred)
                self.w += (x.T * f_)
                self.b += f_[0]
                self.u += cnt * (x.T * f_)
                self.beta += cnt * f_[0]
            cnt += 1
        self.w -= self.u / cnt
        self.b -= self.beta / cnt

    def predict(self, x):
        x = np.dot(x, self.w)
        x[:, 0] += self.b[0]
        x[:, 1] += self.b[1]
        y_pred = self.activation_func(x)
        return y_pred

    def activation_func(self, x):
        return np.where(x > 0, 1, 0)


model2 = Perception2()
model2.fit(features, target)

w = np.array(model2.w).tolist()
b = np.array(model2.b).tolist()
params = {
    "w": w,
    "b": b,
    "dic": dic,
    "count": count,
    "word2index": word2index,
    "words": list(words)
}
with open("averagedmodel.txt", "w") as f:
    json.dump(params, f)

print("123123123")