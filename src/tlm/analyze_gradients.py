import pickle
import numpy as np
from path import output_path
import os

def load_and_analyze_gradient():
    p = os.path.join(output_path, "grad.pickle")
    r = pickle.load(open(p, "rb"))

    p = os.path.join(output_path, "logits.pickle")
    logit = pickle.load(open(p, "rb"))
    analyze_gradient(r, logit)


def load_and_analyze_hv():
    p = os.path.join(output_path, "hv_tt.pickle")
    hv_tt = pickle.load(open(p, "rb"))

    p = os.path.join(output_path, "hv_lm.pickle")
    hv_lm = pickle.load(open(p, "rb"))

    p = os.path.join(output_path, "grad.pickle")
    tt_grad = pickle.load(open(p, "rb"))

    analyze_hv(hv_tt, hv_lm, tt_grad)



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
    for layers, emb in hv:
        combined = np.stack([emb]+layers)
        l.append(combined)
    r = np.concatenate(l, axis=1)
    r = np.transpose(r, [1,0,2,3])
    return r



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

        if abs(g[i]) > 1e-2:
            if d < 1e-1:
                n_diff_1 += 1
            if d < 1e-2:
                n_diff_2 += 1


    return n_diff_1, n_diff_2



def analyze_hv(hv_tt, hv_lm, tt_grad):
    batch_size = 16
    seq_len = 200
    hidden_dim = 768
    reshaped_grad = reshape_gradienet(tt_grad, seq_len, hidden_dim, False)

    hv_tt = reshape(hv_tt)
    hv_lm = reshape(hv_lm)

    assert len(hv_lm) == len(hv_tt)
    for inst_i in range(len(hv_lm)):
        for layer_i in range(13):
            layer_no = layer_i
            if layer_no >= 1:
                print("Layer {} :".format(layer_no), end=" ")
            else:
                print("Embedding:", end=" ")
            for seq_i in range(seq_len):
                n_diff_1, n_diff_2 = diff_and_grad(hv_lm[inst_i, layer_i, seq_i],
                                                 hv_tt[inst_i, layer_i, seq_i],
                                                 reshaped_grad[inst_i, layer_i, seq_i])
                print("{}({})".format(n_diff_1, n_diff_2), end=" ")
            print("\n")
        print("-----------------")


if __name__ == '__main__':
    load_and_analyze_hv()

