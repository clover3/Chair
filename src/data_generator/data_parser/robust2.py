import os

import cpath
from galagos.basic import load_galago_ranked_list


def load_2k_rank():
    path =os.path.join(robust_path, "rob04.desc.galago.2k.out")
    f = open(path, "r")

    ranked_list = {}

    for line in f:
        q_id, _, doc_id, rank, score, _ = line.split()

        if q_id not in ranked_list:
            ranked_list[q_id] = []

        ranked_list[q_id].append((doc_id, int(rank), score))

    return ranked_list


def load_bm25_best():
    path =os.path.join(robust_path, "rob04.desc.galago.2k.out")
    return load_galago_ranked_list(path)

def load_qrel(path):
    f = open(path, "r")
    q_dict = {}
    for line in f:
        q_id, _, doc_id, score = line.split()
        if q_id not in q_dict:
            q_dict[q_id] = {}

        q_dict[q_id][doc_id] = int(score)

    return q_dict


def load_robust_qrel():
    qrel_path = os.path.join(robust_path, "qrels.rob04.txt")
    return load_qrel(qrel_path)



def select_top_k_galago(k):
    path =os.path.join(robust_path, "rob04.desc.galago.2k.out")
    f = open(path, "r")

    all_lines = []

    for line in f:
        q_id, _, doc_id, rank, score, _ = line.split()

        if int(rank) <= k:
            all_lines.append(line)

    return all_lines

robust_path = os.path.join(cpath.data_path, "robust")