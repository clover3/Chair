import pickle
import numpy as np
from path import output_path
import os

def load_and_analyze():
    p = os.path.join(output_path, "grad.pickle")
    r = pickle.load(open(p, "rb"))
    analyze(r)



def count_non_zeroes(v):
    return count_larger_than(v, 1e-3)

def count_larger_than(v, threshold):
    return np.sum(np.less(threshold, np.abs(v)))


def reshape_gradienet(r, seq_len, hidden_dim):
    reshaped_grad = []
    for batch in r:
        reform = []
        for layer in batch:
            grad = layer[0]
            grad = np.reshape(grad, [-1, seq_len, hidden_dim])
            reform.append(grad)

        reform = reform[-1:] + reform[:-1]
        reform = reform[::-1]
        t = np.stack(reform)
        reshaped_grad.append(t)

    reshaped_grad = np.concatenate(reshaped_grad, axis=1) # [num_layers, num_insts, seq_len, hidden_dims]
    reshaped_grad = np.transpose(reshaped_grad, [1,0,2,3])
    return reshaped_grad


def analyze(r):
    batch_size = 16
    seq_len = 200
    hidden_dim = 768
    reshaped_grad = reshape_gradienet(r, seq_len, hidden_dim)
    print(reshaped_grad.shape)

    for inst_i in range(len(reshaped_grad)):
        for layer_i in range(13):
            layer_no = 12 - layer_i
            if layer_no >= 1:
                print("Layer {} :".format(layer_no), end=" ")
            else:
                print("Embedding:", end=" ")
            for seq_i in range(seq_len):
                v = reshaped_grad[inst_i, layer_i, seq_i]
                print("{}/{}/{}".format(count_larger_than(v, 1e-1), count_larger_than(v, 1e-2),  count_non_zeroes(v)), end=" ")
            print("\n")
        print("-----------------")
if __name__ == '__main__':
    load_and_analyze()
