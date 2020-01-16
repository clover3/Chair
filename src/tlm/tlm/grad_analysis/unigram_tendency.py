import pickle
import sys
from collections import Counter

import math

from cie.arg.kl import kl_divergence
from misc_lib import get_dir_files
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.unigram_lm_from_tfrecord import LM


def dir_itr(dir_path):
    for file_path in get_dir_files(dir_path):
        data = EstimatorPredictionViewer(file_path)
        for entry in data:
            yield entry

class LMClassifer:
    def __init__(self, tf_true, tf_false):
        self.true_ctf = sum(tf_true.values())
        self.tf_true = tf_true
        self.false_ctf = sum(tf_false.values())
        self.tf_false = tf_false

    def term_odd(self, token):
        smoothing = 0.001
        P_t_True = self.tf_true[token] / self.true_ctf
        P_t_False = self.tf_false[token] / self.false_ctf

        P_t_True_s = P_t_True * (1-smoothing) + P_t_False * smoothing
        P_t_False_s = P_t_False * (1 - smoothing) + P_t_True * smoothing
        log_true = math.log(P_t_True_s)
        log_false = math.log(P_t_False_s)
        return log_true - log_false

    def term_contrib(self, token):
        return self.term_odd(token) * self.tf_true[token]

def get_bigrams(tokens):
    prev_t = ""
    for t in tokens:
        yield prev_t + "_" + t
        prev_t = t


class NormFn:
    def __init__(self, t):
        self.t = t
        self.name = "NormFn {}".format(t)

    def __call__(self, score, loss):
        norm_score = score / loss
        return norm_score > self.t

class ScoreFn:
    def __init__(self, t):
        self.t = t
        self.name = "ScoreFn {}".format(t)

    def __call__(self, score, loss):
        return score > self.t

def main(path):
    cls_fn_list = []

    for t in [3500, 4500 ]:
        cls_fn_list.append(NormFn(t) )
    for t in [3500, 4500 ]:
        cls_fn_list.append(ScoreFn(t) )

    lm_dict = {}
    for idx, label_fn in enumerate(cls_fn_list):
        lm_dict[(idx, True)] = LM(True)
        lm_dict[(idx, False)] = LM(True)

    inst_cnt = Counter()
    for entry in dir_itr(path):
        score = entry.get_vector("overlap_score")
        masked_lm_example_loss = entry.get_vector("masked_lm_example_loss")
        tokens = entry.get_tokens("input_ids")

        tokens = get_bigrams(tokens)
        lm = LM(True)
        lm.update(tokens)
        if masked_lm_example_loss == 0:
            continue

        for idx, label_fn in enumerate(cls_fn_list):
            label = label_fn(score, masked_lm_example_loss)
            inst_cnt[(idx, label)] += 1
            lm_dict[(idx, label)].update_from_lm(lm)

    entries = []
    for idx, fn_obj in enumerate(cls_fn_list):
        true_lm = lm_dict[(idx, True)]
        false_lm = lm_dict[(idx, False)]
        obj = idx, fn_obj, true_lm, false_lm
        entries.append(obj)

    data = entries , inst_cnt
    pickle.dump(data, open("TU.pickle", "wb"))
    print_info(entries, inst_cnt)


def print_info(entries, inst_cnt):
    for idx, fn_obj, true_lm, false_lm in entries:
        lm_cls = LMClassifer(true_lm.tf, false_lm.tf)
        print("{} {}".format(idx, fn_obj.name))
        print("True {} / False {}".format(inst_cnt[(idx, True)], inst_cnt[(idx, False)]))
        print("kl_div : ", kl_divergence(true_lm.tf, false_lm.tf))

        contribs = []
        terms = set(list(true_lm.tf.keys()) + list(false_lm.tf.keys()))
        for t in terms:
            c = lm_cls.term_contrib(t)
            o = lm_cls.term_odd(t)
            contribs.append((t, c, o))

        contribs.sort(key=lambda x: x[1], reverse=True)

        show_num = 20
        print("< True >")
        for t, c, o in contribs[:show_num]:
            print("{0} {1:.2f} {2:.2f}".format(t, c, o))
        print("< False> ")
        for t, c, o in contribs[::-1][:show_num]:
            print("{0} {1:.2f} {2:.2f}".format(t, c, o))
        print("")


def show_from_pickle():
    data = pickle.load(open("TU.pickle", "rb"))
    entries, inst_cnt = data
    print_info(entries, inst_cnt)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        show_from_pickle()



