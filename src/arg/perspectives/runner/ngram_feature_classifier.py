from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer

from arg.perspectives.cpid_def import CPID
from arg.perspectives.n_gram_feature_collector import PCVectorFeature
from cache import load_from_pickle, save_to_pickle
from list_lib import lmap


def feature_analysis(feature_size, pred_y, avg_x, model, train, y):
    t = np.multiply(avg_x, model.coef_)
    contrib = t[0]
    ranked_idx = np.argsort(contrib)

    selected_ngram_set = load_from_pickle("selected_ngram_feature")

    def get_feature_name(idx):
        n = int(idx / 100) + 1 + 1
        sub_idx = idx % 100
        return selected_ngram_set[n][sub_idx]

    def print_feature_at(idx):
        print(get_feature_name(idx))

    print("Top k features (POS)")
    for i in range(30):
        idx = ranked_idx[i]
        print(get_feature_name(idx), contrib[idx], model.coef_[0][idx])
    print("Top k features (NEG)")
    for i in range(30):
        j = feature_size - 1 - i
        idx = ranked_idx[j]
        print(get_feature_name(idx), contrib[idx], model.coef_[0][idx])
    print("In training data")
    print("pred\tgold\tterms")
    for i in range(100):
        v = train[i].vector

        important_ones = []
        for i in range(feature_size):
            if v[i]:
                score = model.coef_[0][i] * v[i]
                e = get_feature_name(i), score
                important_ones.append(e)

        important_ones.sort(key=lambda x: abs(x[1]), reverse=True)

        terms = ""
        for name, cnt in important_ones[:5]:
            terms += "{0}({1:.2f}) ".format(name, cnt)

        print(pred_y[i], y[i], terms)  #


def ngram_logit():
    train: List[PCVectorFeature] = load_from_pickle("pc_ngram_all_train_ngram_features")
    val: List[PCVectorFeature] = load_from_pickle("pc_ngram_all_ngram_features")

    feature_size = len(train[0].vector) - 100

    def featurize(pc_vector_feature: PCVectorFeature):
        y = int(pc_vector_feature.claim_pers.label)
        v = np.array(pc_vector_feature.vector)
        v = v[100:]
        return v, int(y)

    raw_x, y = zip(*lmap(featurize, train))
    raw_val_x, val_y = zip(*lmap(featurize, val))
    transformer = Normalizer().fit(raw_x)
    x = transformer.transform(raw_x)
    val_x = transformer.transform(raw_val_x)

    model = LogisticRegression()
    #model = LinearSVC()
    model.fit(x, y)

    x_a = np.array(x)
    print(x_a.shape)

    def acc(y, pred_y):
        return np.average(np.equal(y, pred_y))

    pred_y = model.predict(x)
    print("train acc", acc(y, pred_y))
    val_prediction = model.predict(val_x)
    print("val acc", acc(val_y, val_prediction))

    #save_dev_scores(model, val, val_x)

    avg_x = np.sum(x_a, axis=0)
    feature_analysis(feature_size, pred_y, avg_x, model, train, y)


def save_dev_scores(model, val, val_x):
    val_probs = model.predict_proba(val_x)
    score_d = {}
    for pc_vector_feature, prediction in zip(val, val_probs):
        cid = pc_vector_feature.claim_pers.cid
        pid = pc_vector_feature.claim_pers.pid
        cpid = CPID("{}_{}".format(cid, pid))
        score_d[cpid] = prediction[1]
    save_to_pickle(score_d, "pc_ngram_logits")


if __name__ == "__main__":
    ngram_logit()
