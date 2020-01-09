from collections import Counter

import numpy as np
from tensorflow.python import pywrap_tensorflow

from cpath import get_bert_full_path
from visualize.html_visual import HtmlVisualizer, Cell


def load_param(model_path):
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    d_map = reader.get_variable_to_dtype_map()
    numpy_map = {}
    for key in d_map:
        data = reader.get_tensor(key)
        numpy_map[key] = data

    return numpy_map


class BinCounter:
    def __init__(self, ranges):
        self.ranges = ranges
        self.count = Counter()

    def add(self, number):
        for st, ed in self.ranges:
            if st < number < ed:
                self.count[(st, ed)] += 1
                break


def numpy_bins(ndarr, ranges):
    d = {}
    for st, ed in ranges:
        f1 = np.less_equal(st, ndarr)
        f2 = np.less(ndarr, ed)
        t =np.logical_and(f1, f2)
        n = np.count_nonzero(t)
        d[st,ed] = n
    return d


def analyze_parameter():
    p_base = load_param(get_bert_full_path())
    nli_path = "C:\work\Code\Chair\output\model\\runs\\nli_model.ckpt-75000_NLI\\model-0"
    p_ft = load_param(nli_path)
    keys = list(p_base.keys())
    keys.sort()
    infos = {}

    for key in keys:
        print(key)
        if "cls/" in key:
            continue

        param1 = p_base[key]
        param2 = p_ft[key]
        max1 = np.max(np.abs(param1))
        max2 = np.max(np.abs(param2))

        if len(param1.shape) > 1:
            col_max1 = np.max(np.sum(np.abs(param1), axis=0))
            col_max2 = np.max(np.sum(np.abs(param2), axis=0))
        else:
            col_max1 = -1
            col_max2 = -1
        p_avg = np.average(np.abs(param1))
        p_avg2 = np.average(np.abs(param2))
        p_std = np.std(param1)
        print(key, param1.shape, p_avg, p_std)
        cut_off = abs(p_avg) * 0.5
        if cut_off < 1e-2:
            ranges = [(0, cut_off),
                      (cut_off, 1e-2),
                      (1e-2, 1e-1),
                      (1e-1, 999)
                      ]
        else:
            ranges = [(0, cut_off),
                      (cut_off, 999)]

        diff = np.abs(param2 - param1)
        d = numpy_bins(diff, ranges)
        almost_no_change = d[ranges[0]]
        no_change_rate = almost_no_change / diff.size
        if key.startswith("bert/encoder"):
            key_tokens = key.split("/")
            layer_name = key_tokens[2]
            post_fix = "/".join(key_tokens[3:])

            if post_fix not in infos:
                infos[post_fix] = {}
            infos[post_fix][layer_name] = {
                "no_change": no_change_rate,
                "avg": cut_off,
                "std":p_std,
                "max1":max1,
                "max2":max2,
                "p_avg":p_avg,
                "p_avg2":p_avg2,
                "col_max1": col_max1,
                "col_max2": col_max2
            }
        print(key, no_change_rate)

        for key, value in d.items():
            print(key, value)

    for post_fix in infos:
        print(post_fix)
        for layer_i in range(12):
            layer_name = "layer_{}".format(layer_i)
            cur_info = infos[post_fix][layer_name]
            print(layer_name, cur_info["no_change"], cur_info["col_max1"], cur_info["col_max2"])

def print_param():
    p_base = load_param(get_bert_full_path())
    nli_path = "C:\work\Code\Chair\output\model\\runs\\nli_model.ckpt-75000_NLI\\model-0"
    p_ft = load_param(nli_path)
    keys = list(p_base.keys())

    key = "bert/encoder/layer_0/output/dense/kernel"
    param1 = p_base[key]
    param2 = p_ft[key]
    html = HtmlVisualizer("bert_dense_param.html")

    l , c = param1.shape

    s_score = 100
    for i in range(l):
        rows = []
        row1 = []
        row2 = []
        s_score = 100 - s_score
        score = s_score
        for j in range(c):
            score = 100 - score
            row1.append(Cell("{0:.4f}".format(param1[i, j]), score))
            row2.append(Cell("{0:.4f}".format(param2[i, j]), score))
        rows.append(row1)
        rows.append(row2)
        html.write_table(rows)



if __name__ == "__main__":
    analyze_parameter()
