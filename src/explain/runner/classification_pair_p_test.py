import os
import pickle

import numpy as np

import cpath
from list_lib import lmap


def load(name):
    f= open(os.path.join(cpath.output_path, name), "rb")
    return pickle.load(f)

def acc(binary_list):
    return np.sum(binary_list) / len(binary_list)

def pair_compare(run_A, run_B):
    pred_A, gold_A = run_A
    pred_B, gold_B = run_B
    assert np.all(np.equal(gold_B, gold_A))

    assert len(pred_A) == len(pred_B)
    assert len(gold_A) == len(gold_B)

    binary_A = np.equal(pred_A, gold_A)
    binary_B = np.equal(pred_B, gold_B)
    binary_A = binary_A.astype(float)
    binary_B = binary_B.astype(float)
    print(acc(binary_A))
    print(acc(binary_B))




def pair_test_runner():
    runs = ["nli_pred_nli_from_bert_model-73630",
            "nli_pred_nli_ex_19_model-73630",
            "nli_pred_nli_ex_21_model-73630",
            ]

    run_data = lmap(load, runs)

    n_runs = len(runs)
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            run_A = run_data[i]
            run_B = run_data[j]
            print(i,j)
            r = pair_compare(run_A, run_B)
            print(r)



if __name__ == "__main__":
    pair_test_runner()