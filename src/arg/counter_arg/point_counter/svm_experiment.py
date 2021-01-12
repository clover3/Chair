from typing import List, Tuple

import numpy as np
from sklearn.svm import LinearSVC

from arg.counter_arg.header import Passage
from arg.counter_arg.point_counter.prepare import load_data
from cache import save_to_pickle
from list_lib import lmap, right
from misc_lib import tprint
from models.baselines import svm
from task.metrics import accuracy


def main():
    dev_x, dev_y, train_x, train_y = get_data()
    tprint("training and testing")
    use_char_ngram = False
    print("Use char ngram", use_char_ngram )
    pred_svm_ngram = svm.train_svm_and_test(svm.NGramFeature(use_char_ngram, 4), train_x, train_y, dev_x)
    # pred_svm_ngram = list([random.randint(0,1) for _ in dev_y])
    result = accuracy(pred_svm_ngram, dev_y)
    print(result)


def test_set_importance(coef, x_test, y_label):
    importance = np.zeros_like(coef)
    n_iter = 0
    for x, y in zip(x_test, y_label):
        for i, j in zip(*x.nonzero()):
            cnt = x[i, j]
            idx = j
            # sec_w = max([coef[l, idx] for l in range(2) if l != y])
            sec_w = 0
            w = coef[0, idx] - sec_w
            importance[0, idx] += w * cnt
        n_iter += 1
    return importance


def show_features():
    dev_x, dev_y, train_x, train_y = get_data()
    tprint("training and testing")
    feature_extractor = svm.NGramFeature(False, 4)
    X_train_counts = feature_extractor.fit_transform(train_x)
    save_to_pickle(feature_extractor, "feature_extractor")

    svclassifier = LinearSVC()
    svclassifier.fit(X_train_counts, train_y)
    save_to_pickle(svclassifier, "svclassifier")
    # svclassifier = load_from_pickle("svclassifier")
    print(svclassifier.coef_)
    print(svclassifier.coef_.shape)
    word_feature_names = feature_extractor.word_feature.get_feature_names()
    char_feature_dict = {v: k for k, v in feature_extractor.char_feature_dict.items()}

    base = len(word_feature_names) + 1

    def get_feature_name(j):
        if j < len(word_feature_names):
            return "word: " + str(word_feature_names[j])
        else:
            return "char: " + str(char_feature_dict[j - base])

    importance = test_set_importance(svclassifier.coef_, X_train_counts, dev_y)
    save_to_pickle(importance, "importance")
    # importance = load_from_pickle("importance")
    label = 0
    w = importance[label]
    print("Label ", label)

    def print_feature(j):
        feature_name = get_feature_name(j)
        print("{} {} {} {}".format(j, feature_name, svclassifier.coef_[0, j], w[j]))

    for j in np.argsort(w)[::-1][:60]:
        print_feature(j)
    print("---")
    for j in np.argsort(w)[:60]:
        print_feature(j)


def get_data():
    tprint("Loading data")
    train_data: List[Tuple[Passage, int]] = load_data("training")
    dev_data = load_data("validation")

    def get_texts(e: Tuple[Passage, int]) -> str:
        return e[0].text.replace("\n", " ")

    train_x = lmap(get_texts, train_data)
    train_y = right(train_data)
    dev_x = lmap(get_texts, dev_data)
    dev_y = right(dev_data)
    return dev_x, dev_y, train_x, train_y


# [{'precision': 0.920751633986928, 'recall': 0.8756798756798757, 'f1': 0.8976503385105535},
# {'precision': 0.8814814814814815, 'recall': 0.9246309246309247, 'f1': 0.9025407660219947}]

if __name__ == "__main__":
    show_features()
    # main()
#
