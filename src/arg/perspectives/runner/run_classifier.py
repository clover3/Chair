from collections import Counter
from typing import Dict, List

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

    def fileter_fn(data_point: Dict):
        remove_stopword_and_punct(stopwords, data_point['feature'])

    foreach(fileter_fn, train)

    def is_correct(data_point: Dict):
        x = data_point['feature']
        y = int(data_point['label'])
        return classifier.predict(x) == int(y)

    correctness = lmap(is_correct, val)

    print("val acc: ", average(correctness))


def test_logistic_regression():
    train_and_val = load_feature_and_split()
    train: List[Dict] = train_and_val[0]
    val: List[Dict] = train_and_val[1]
    valid_datapoint_list: List[Dict] = train + val
    stopwords = load_stopwords()

    def fileter_fn(data_point: Dict):
        remove_stopword_and_punct(stopwords, data_point['feature'])
    foreach(fileter_fn, train)
    foreach(fileter_fn, val)

    tf_list = lmap(lambda dp: dp['feature'], valid_datapoint_list)
    tf_acc = Counter()
    for tf in tf_list:
        tf_acc.update(tf)

    voca: List[str] = left(tf_acc.most_common(10000))
    #voca = set(flatten(lmap(get_voca_from_datapoint, valid_datapoint_list)))
    voca2idx: Dict[str, int] = dict(zip(list(voca), range(len(voca))))
    idx2voca: Dict[int, str] = {v: k for k, v in voca2idx.items()}
    print("Num voca:", len(voca))
    feature_size = len(voca) + 1

    def featurize(datapoint: Dict):
        tf = datapoint['feature']
        y = int(datapoint['label'])
        v = np.zeros([feature_size])
        for t, prob in tf.items():
            if t in voca2idx:
                v[voca2idx[t]] = prob
        v[-1] = datapoint['num_mention']
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
    contrib = t[0]
    ranked_idx = np.argsort(contrib)
    def print_feature_at(idx):
        if idx == feature_size -1 :
            print("[NUM_MENTION]", contrib[idx])
        else:
            print(idx2voca[idx], contrib[idx])

    print("Top k features (POS)")
    for i in range(30):
        idx = ranked_idx[i]
        print_feature_at(idx)

    print("Top k features (NEG)")
    for i in range(30):
        j = len(voca) -1 -i
        idx = ranked_idx[j]
        print_feature_at(idx)

    print("In training data")
    print("pred\tgold\tterms")
    for i in range(100):
        terms = left(train[i]['feature'].most_common(50))
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
    test_logistic_regression()
