import random
import sys

import math
import numpy as np

from misc_lib import average
from tf_util.enum_features import load_record2


def proportion_random(prob1):
    pred_prob2 = random.random() * prob1
    return pred_prob2

def same(x):
    return x

def independent_model(loss1, loss2, mask, method):
    loss_list = []
    for y1, y2 in zip(loss1, loss2):
        gold_prob1 = math.exp(-y1)
        gold_prob2 = math.exp(-y2)

        pred_prob1 = gold_prob1

        pred_prob2 = method(pred_prob1)


        def cross_entropy(pred_prob, gold_prob):
            v = - pred_prob * math.log(gold_prob) - (1-pred_prob) * math.log(1-gold_prob1+0.0000001)
            return v

        loss1 = cross_entropy(pred_prob1, gold_prob1)
        loss2 = cross_entropy(pred_prob2, gold_prob2)

        loss = loss1 + loss2

        loss_list.append(loss)

    return np.sum(np.array(loss_list) * mask)

def diff_model(loss1, loss2, mask):
    loss_list = []
    for y1, y2 in zip(loss1, loss2):
        gold_prob1 = math.exp(-y1)
        gold_prob2 = math.exp(-y2)

        gold = gold_prob1 - gold_prob2

        loss = abs(gold - random.random())
        loss_list.append(loss)
    return np.sum(np.array(loss_list) * mask)


def run(filename, n_item):
    loss_list = []
    loss_list2 = []
    loss_list3 = []
    for idx, features in enumerate(load_record2(filename)):
        if idx > n_item :
            break
        keys = features.keys()
        loss1 = features["loss_base"].float_list.value
        loss2 = features["loss_target"].float_list.value
        mask = features["masked_lm_weights"].float_list.value


        loss_list.append(independent_model(loss1, loss2, mask, proportion_random))
        loss_list2.append(diff_model(loss1, loss2, mask))
        loss_list3.append(independent_model(loss1, loss2, mask, same))

    print("independent (proportion random): ", average(loss_list))
    print("diff  : ", average(loss_list2))
    print("independent (same): ", average(loss_list3))


if __name__ == "__main__":
    filename = sys.argv[1]
    if len(sys.argv) == 3:
        n_item = int(sys.argv[2])
    else:
        n_item = 1

    run(filename, n_item)





