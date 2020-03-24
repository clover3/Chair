import os
from functools import partial

import numpy as np
from sklearn.linear_model import LogisticRegression

from arg.perspectives.basic_analysis import load_train_data_point
from list_lib import lmap, lmap_w_exception, lfilter, left, flatten, right
from sydney_clueweb.clue_path import index_name_list


def featurize_fn(voca, voca2idx, datapoint):
    rm_list, label = datapoint
    nonzero = lfilter(lambda x: x > 0, right(rm_list))
    if nonzero:
        nonzero_min = min(nonzero)
    else:
        nonzero_min = 0

    terms = left(rm_list)
    term_ids = lmap(lambda x: voca2idx[x], terms)
    scores = list([s if s > 0 else 0.2 * nonzero_min for s in right(rm_list)])

    v = np.zeros([len(voca)])
    for idx, score in zip(term_ids, scores):
        v[idx] = score
    return v, label


def test_rm_classifier():
    datapoint_list = load_train_data_point()

    disk_name = index_name_list[0]
    dir_path = "/mnt/nfs/work3/youngwookim/data/perspective/train_claim_perspective/rm3"

    def get_rm(data_point):
        label, cid, pid, claim_text, p_text = data_point
        file_name = "{}_{}_{}.txt".format(disk_name, cid, pid)
        f = open(os.path.join(dir_path, file_name))

        def parse_line(line):
            term, prob = line.split("\t")  #
            prob = float(prob) * 1000
            return term, prob

        return lmap(parse_line, f), int(label)

    valid_datapoint_list = lmap_w_exception(get_rm, datapoint_list, FileNotFoundError)
    print("Total of {} data point".format(len(valid_datapoint_list)))

    voca = set(left(flatten(left(valid_datapoint_list))))
    voca2idx = dict(zip(list(voca), range(len(voca))))
    idx2voca = {v: k for k, v in voca2idx.items()}

    split = int(len(valid_datapoint_list) * 0.7)
    train_data = valid_datapoint_list[:split]
    val_data = valid_datapoint_list[split:]


    pos_data = lfilter(lambda x: x[1] == "1", valid_datapoint_list)
    neg_data = lfilter(lambda x: x[1] == "0", valid_datapoint_list)
    featurize = partial(featurize_fn, voca, voca2idx)
    x, y = zip(*lmap(featurize, train_data))
    val_x, val_y = zip(*lmap(featurize, val_data))

    model = LogisticRegression()
    model.fit(x, y)

    x_a = np.array(x)
    print(x_a.shape)
    avg_x = np.sum(x_a, axis=0)

    contrib = np.multiply(avg_x, model.coef_)[0]
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


    def acc(y, pred_y):
        return np.average(np.equal(y, pred_y))

    pred_y = model.predict(x)
    print("train acc", acc(y, pred_y))
    print("val acc", acc(val_y, model.predict(val_x)))


if __name__ == "__main__":
    test_rm_classifier()