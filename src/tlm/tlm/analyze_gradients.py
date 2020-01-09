import os
import pickle
from collections import Counter

import numpy as np

from cpath import output_path, data_path
from data_generator import tokenizer_wo_tf
from visualize.html_visual import HtmlVisualizer, Cell


def load_and_analyze_gradient():
    p = os.path.join(output_path, "grad.pickle")
    r = pickle.load(open(p, "rb"))

    p = os.path.join(output_path, "logits.pickle")
    logit = pickle.load(open(p, "rb"))
    analyze_gradient(r, logit)


def load_and_analyze_hv():
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))


    p = os.path.join(output_path, "hv_tt.pickle")
    hv_tt = pickle.load(open(p, "rb"))

    p = os.path.join(output_path, "hv_lm.pickle")
    hv_lm = pickle.load(open(p, "rb"))

    p = os.path.join(output_path, "grad.pickle")
    tt_grad = pickle.load(open(p, "rb"))

    analyze_hv(hv_tt, hv_lm, tt_grad, tokenizer)



def count_non_zeroes(v):
    return count_larger_than(v, 1e-3)

def count_larger_than(v, threshold):
    return np.sum(np.less(threshold, np.abs(v)))


def reshape_gradienet(r, seq_len, hidden_dim, reverse=True):
    reshaped_grad = []
    for batch in r:
        reform = []
        for layer in batch:
            grad = layer[0]
            grad = np.reshape(grad, [-1, seq_len, hidden_dim])
            reform.append(grad)

        reform = reform[-1:] + reform[:-1]
        if reverse:
            reform = reform[::-1]
        t = np.stack(reform)
        reshaped_grad.append(t)

    reshaped_grad = np.concatenate(reshaped_grad, axis=1) # [num_layers, num_insts, seq_len, hidden_dims]
    reshaped_grad = np.transpose(reshaped_grad, [1,0,2,3])
    return reshaped_grad


def analyze_gradient(r, logit):
    logits = np.concatenate(logit, axis=0)
    print(logits.shape)
    batch_size = 16
    seq_len = 200
    hidden_dim = 768
    reshaped_grad = reshape_gradienet(r, seq_len, hidden_dim)
    print(reshaped_grad.shape)

    for inst_i in range(len(reshaped_grad)):
        label = np.argmax(logits[inst_i])
        print("Label : {}\t({})\t".format(label, logits[inst_i]))
        for layer_i in range(13):
            layer_no = 12 - layer_i
            if layer_no >= 1:
                print("Layer {} :".format(layer_no), end=" ")
            else:
                print("Embedding:", end=" ")
            for seq_i in range(seq_len):
                v = reshaped_grad[inst_i, layer_i, seq_i]
                print("{}/{}".format(count_larger_than(v, 1e-1), count_larger_than(v, 1e-2)), end=" ")
            print("\n")
        print("-----------------")


def reshape(hv):
    l = []
    x_list = []
    for layers, emb, x0 in hv:
        combined = np.stack([emb]+layers)
        l.append(combined)
        x_list.append(x0)
    r = np.concatenate(l, axis=1)
    r = np.transpose(r, [1,0,2,3])
    x_list = np.concatenate(x_list, axis=0)
    return r, x_list



def diff_vals(v1, v2):
    l = len(v1)

    n_diff_1 = 0
    n_diff_2 = 0

    n_v1_zero = 0
    n_v2_zero = 0

    assert len(v1) == len(v2)
    for i in range(l):
        d = np.abs(v1[i]-v2[i])

        if d < 1e-1:
            n_diff_1 += 1
        if d < 1e-2:
            n_diff_2 += 1

        if np.abs(v1[i]) < 1e-2:
            n_v1_zero += 1
        if np.abs(v2[i]) < 1e-2:
            n_v2_zero += 1

    return n_diff_1, n_diff_2, n_v1_zero, n_v2_zero


def diff_and_grad(v1, v2, g):
    l = len(v1)

    n_diff_1 = 0
    n_diff_2 = 0

    assert len(v1) == len(v2)
    for i in range(l):
        d = np.abs(v1[i]-v2[i])

        if abs(g[i]) > 1e-3 :
            if d < 1e-1:
                n_diff_1 += 1
            if d < 10:
                n_diff_2 += 1

    return n_diff_1, n_diff_2


def calculate_diff_prob(hv_tt, hv_lm, tokenizer):
    batch_size = 16
    seq_len = 200
    hidden_dim = 768

    hv_tt, x_list = reshape(hv_tt)
    hv_lm, x_list = reshape(hv_lm)

    count = Counter()
    all_cnt = 0
    assert len(hv_lm) == len(hv_tt)
    n_inst = min(len(hv_lm), 5)
    for inst_i in range(n_inst):
        print(inst_i, end=" ")
        all_cnt += seq_len
        for layer_i in range(13):
            layer_no = layer_i
            for seq_i in range(seq_len):
                v1 = hv_lm[inst_i, layer_i, seq_i]
                v2 = hv_tt[inst_i, layer_i, seq_i]

                diff = np.abs(v1-v2)

                for dim_i in range(len(v1)):
                    if diff[dim_i] < 1e-1:
                        count[(layer_i, dim_i)] += 1
    print()
    for layer_i in range(13):
        p_distrib = Counter()
        for dim_i in range(hidden_dim):
            p = count[(layer_i, dim_i)] / all_cnt
            assert p <= 1
            bin = int((p+0.05) * 10)
            p_distrib[bin] += 1
        print(layer_i, p_distrib)




def analyze_hv(hv_tt, hv_lm, tt_grad, tokenizer):
    batch_size = 16
    seq_len = 200
    hidden_dim = 768
    reshaped_grad = reshape_gradienet(tt_grad, seq_len, hidden_dim, False)

    hv_tt, x_list = reshape(hv_tt)
    hv_lm, x_list = reshape(hv_lm)

    assert len(hv_lm) == len(hv_tt)

    html = HtmlVisualizer("Preserved.html")
    for inst_i in range(len(hv_lm)):
        print("\t", end="")
        tokens = tokenizer.convert_ids_to_tokens(x_list[inst_i])
        for seq_i in range(seq_len):
            token = tokenizer.convert_ids_to_tokens([x_list[inst_i, seq_i]])[0]
            print("{}".format(token), end="\t")
        print()
        scores = []
        for layer_i in range(13):
            if layer_i != 1 :
                continue
            layer_no = layer_i
            if layer_no >= 1:
                print("Layer {} :".format(layer_no), end="\t")
            else:
                print("Embedding:", end="\t")
            for seq_i in range(seq_len):
                n_diff_1, n_diff_2 = diff_and_grad(hv_lm[inst_i, layer_i, seq_i],
                                                 hv_tt[inst_i, layer_i, seq_i],
                                                 reshaped_grad[inst_i, layer_i, seq_i])
                scores.append(n_diff_1)
                print("{}({})".format(n_diff_1, n_diff_2), end="\t")
            print("\n")

        row = []
        for t, s in zip(tokens, scores):
            score = s / hidden_dim * 100
            row.append(Cell(t, score))
        html.write_table([row])
        print("-----------------")


if __name__ == '__main__':
    load_and_analyze_hv()

