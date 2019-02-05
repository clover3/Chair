import random
import pickle
import os
import sys

from config.input_path import galago_judge_path
from collections import defaultdict
## read data

def read_list(i):
    st = i * 10000
    ed = (i+1) * 10000
    dir_path = galago_judge_path
    file_path = os.path.join(dir_path, "{}_{}.list".format(st, ed))
    f = open(file_path, "r")
    group_by_q_id = defaultdict(list)
    for line in f:
        q_id, _, doc_id, rank, score, _ = line.split()
        group_by_q_id[q_id].append((doc_id, score))

    return group_by_q_id

## debias sampling
def sample_debias(ranked_list):
    max_occurence = 100
    output = {}
    for doc_id, score in ranked_list:
        score_grouper = int(float(score) + 0.8)
        if score_grouper not in output:
            output[score_grouper] = []
        if len(output[score_grouper]) < max_occurence:
            output[score_grouper].append((doc_id, score))
    return output


def pick1(l):
    return l[random.randrange(len(l))]


def generate_data(run_id):
    group_by_q_id = read_list(run_id)
    result = []
    for q_id in group_by_q_id:
        l = group_by_q_id[q_id]
        groups = sample_debias(l)
        if max(list(groups.keys())) < 15:
            continue

        def sample_one(keys, d):
            key = pick1(keys)
            entry = pick1(d[key])
            return entry

        keys = list(groups.keys())
        data_size = 50

        print("Sampling")
        for i in range(data_size):
            doc_id_1, score1 = sample_one(keys, groups)
            doc_id_2, score2 = sample_one(keys, groups)

            if score1 < score2 :
                result.append((q_id, doc_id_1, q_id, doc_id_2))
            else:
                result.append((q_id, doc_id_2, q_id, doc_id_1))
    pickle.dump(result, open("merger_idpair_{}.pickle".format(run_id), "wb"))


if __name__ == "__main__":
    run_id = int(sys.argv[1])
    generate_data(run_id)
