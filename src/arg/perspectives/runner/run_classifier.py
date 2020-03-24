from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

from arg.perspectives.collection_based_classifier import learn_lm, mention_num_classifier, load_feature_and_split
from list_lib import lmap, foreach, left
from misc_lib import average
from models.classic.stopword import load_stopwords


def remove_stopword_and_punct(stopwords, counter):
    punct = ".,'`\""
    keys = list(counter.keys())
    for key in keys:
        if key in stopwords or key in punct:
            counter.pop(key)


def test_generative_model():
    train, val = load_feature_and_split()
    print("Training lm")
    classifier = learn_lm(train)
    stopwords = load_stopwords()

    def fileter_fn(data_point):
        remove_stopword_and_punct(stopwords, data_point[0][0])

    foreach(fileter_fn, train)

    def is_correct(elem):
        x, y = elem
        x = x[0]
        return classifier.predict(x) == int(y)

    correctness = lmap(is_correct, val)

    print("val acc: ", average(correctness))

def test_logistic_regression():
    train, val = load_feature_and_split()
    valid_datapoint_list = train + val

    def get_voca_from_datapoint(data_point):
        (tf, num), y = data_point
        return tf.keys()

    tf_list = left(left(valid_datapoint_list))
    tf_acc = Counter()
    for tf in tf_list:
        tf_acc.update(tf)

    voca = left(tf_acc.most_common(10000))
    #voca = set(flatten(lmap(get_voca_from_datapoint, valid_datapoint_list)))
    voca2idx = dict(zip(list(voca), range(len(voca))))
    idx2voca = {v: k for k, v in voca2idx.items()}
    print(len(voca))

    def featurize(datapoint):
        (tf, num), y = datapoint
        v = np.zeros([len(voca)])
        for t, prob in tf.items():
            if t in voca2idx:
                v[voca2idx[t]] = prob
        return v, int(y)

    x, y = zip(*lmap(featurize, train))
    val_x, val_y = zip(*lmap(featurize, val))

    model = LogisticRegression()
    model.fit(x, y)

    x_a = np.array(x)
    print(x_a.shape)
    avg_x = np.sum(x_a, axis=0)

    def acc(y, pred_y):
        return np.average(np.equal(y, pred_y))

    pred_y = model.predict(x)
    print("train acc", acc(y, pred_y))
    print("val acc", acc(val_y, model.predict(val_x)))
    t = np.multiply(avg_x, model.coef_)
    print(t.shape)
    contrib = t[0]
    print(contrib.shape)
    ranked_idx = np.argsort(contrib)
    print(ranked_idx.shape)
    for i in range(30):
        idx = ranked_idx[i]
        print(idx2voca[idx], contrib[idx])

    for i in range(30):
        j = len(voca) -1 -i
        idx = ranked_idx[j]
        print(idx2voca[idx], contrib[idx])

    for i in range(30):
        terms = left(train[i][0][0].most_common(50))
        terms = list(terms[25:])
        print(pred_y[i], y[i], terms) #


def lm_contribution():
    train, val = load_feature_and_split()
    print("Training lm")
    stopwords = load_stopwords()

    def fileter_fn(data_point):
        remove_stopword_and_punct(stopwords, data_point[0][0])

    foreach(fileter_fn, train)
    classifier = learn_lm(train)

    acc_contrib = Counter()
    for data_point in train:
        (tf, num), y = data_point

        contrib = classifier.counter_contribution(tf)
        # print("{} {} {}".format(y, classifier.predict(tf), classifier.counter_odd(tf)))
        # print("--------------")
        for t, score in contrib.most_common(100):
            acc_contrib[t] += score

    for t, score in acc_contrib.most_common(100):
        print(t, score, classifier.P_w_C_dict[t], classifier.P_w_NC_dict[t])

def mention_num_based():
    train, val = load_feature_and_split()
    mention_num_classifier(train, val)


if __name__ =="__main__" :
    mention_num_based()
