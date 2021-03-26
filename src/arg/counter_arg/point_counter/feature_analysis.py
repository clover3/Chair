import numpy as np
from sklearn.svm import LinearSVC

from cache import save_to_pickle
from misc_lib import tprint
from models.baselines import svm


def show_features_by_svm(train_x, train_y):
    tprint("show_features")
    feature_extractor = svm.NGramFeature(False, 4)
    X_train_counts = feature_extractor.fit_transform(train_x)
    # save_to_pickle(feature_extractor, "feature_extractor")
    svclassifier = LinearSVC()
    svclassifier.fit(X_train_counts, train_y)
    save_to_pickle(svclassifier, "svclassifier")
    # svclassifier = load_from_pickle("svclassifier")
    print(svclassifier.coef_.shape)
    word_feature_names = feature_extractor.word_feature.get_feature_names()
    char_feature_dict = {v: k for k, v in feature_extractor.char_feature_dict.items()}
    base = len(word_feature_names) + 1

    def get_feature_name(j):
        if j < len(word_feature_names):
            return "word: " + str(word_feature_names[j])
        else:
            return "char: " + str(char_feature_dict[j - base])

    importance = test_set_importance(svclassifier.coef_, X_train_counts, train_y)
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
