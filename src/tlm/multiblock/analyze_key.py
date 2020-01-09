import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from cpath import data_path, output_path
from data_generator.tokenizer_wo_tf import FullTokenizerWarpper, pretty_tokens
from misc_lib import average, flatten


def plot_data(summary_list):
    fig, ax_l = plt.subplots(5)


    for label in range(5):
        names, data = zip(*summary_list[label])
        ax_l[label].bar(names, data)
        ax_l[label].set_title('label {}'.format(label))

    plt.show()


def test_set_importance(coef, x_test, y_label):
    importance = np.zeros_like(coef)
    for x, y in zip(x_test, y_label):
        for i, j in zip(*x.nonzero()):
            cnt = x[i, j]
            idx = j
            sec_w = max([coef[l, idx] for l in range(5) if l != y])
            w = coef[y,idx] - sec_w


            importance[y, idx] += w * cnt
    return importance



def feature_analysis(data):
    key_list, tokens_list = zip(*list(data))

    subtoken = True
    if subtoken :
        def dummy(doc):
            return doc
        cv = CountVectorizer(
            tokenizer=dummy,
            preprocessor=dummy,
        )
    else:
        tokens_list = list([pretty_tokens(t) for t in tokens_list])
        cv = CountVectorizer()

    X = cv.fit_transform(tokens_list)

    stat = Counter(key_list)
    n_total = len(key_list)

    train_size = int(len(key_list) * 0.8)
    X_train = X[:train_size]
    y_train = key_list[:train_size]
    X_test = X[train_size:]
    y_test = key_list[train_size:]

    model = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("acc", model.score(X_test, y_test))
    print(classification_report(y_pred=y_pred, y_true=y_test))
    coef = model.coef_
    print(coef.shape)
    feature_names = cv.get_feature_names()

    importance = test_set_importance(coef, X_test, y_test)


    print("Importance")
    summary_list = []
    for label in range(5):
        summary = []
        w = importance[label]
        print("Label ", label)
        print(stat[label]/ n_total * 100 , "%")
        for j in np.argsort(w)[::-1][:60]:
            print(feature_names[j], end= " / ")
            summary.append((feature_names[j], w[j]))
        print("")
        summary_list.append(summary[:15])
    plot_data(summary_list)




def get_tokens():
    voca_path = os.path.join(data_path, "bert_voca.txt")
    tokenizer = FullTokenizerWarpper(voca_path)

    file_path = os.path.join(output_path, "multiblock", "argmax1.pickle")

    predictions = pickle.load(open(file_path, "rb"))

    for d in predictions:
        key = d['keys']
        ids = d['input_ids']
        tokens = tokenizer.decode_list(ids)

        yield key, tokens

def text_print():
    d = get_tokens()
    for key, tokens in d:
        print(key)
        print(pretty_tokens(tokens))

def per_label_output():
    d = get_tokens()

    data = list([list() for _ in range(5)])
    for key, tokens in d:
        data[key].append(pretty_tokens(tokens))

    for label in range(5):
        f = open(os.path.join(output_path, "multiblock", "{}.txt".format(label)), "w", encoding="utf-8")
        for text in data[label][:100]:
            f.write(text + "\n")



def text_len():
    d = get_tokens()
    data = list([list() for _ in range(5)])
    for key, tokens in d:
        data[key].append(len(pretty_tokens(tokens)))

    for i in range(5):
        print(i, average(data[i]))

def tf_interactive():
    d = get_tokens()

    data = list([list() for _ in range(5)])
    for key, tokens in d:
        data[key].append(tokens)

    cut = min([len(l) for l in data])

    tf_list = []
    for label in range(5):
        tf = Counter(flatten(data[label][:cut]))
        tf_list.append(tf)

    while True:
        word = input()
        for label in range(5):
            print(label, tf_list[label][word])






def do_feature_analysis():
    feature_analysis(get_tokens())


do_feature_analysis()



