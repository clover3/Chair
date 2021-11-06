import random
from collections import Counter
from typing import List

from cache import load_pickle_from, load_from_pickle
from cpath import at_output_dir
from tlm.qtype.analysis.visualize_q_weights import cluster_rep_by_tf_idf, cluster_q_term_counts
from tlm.qtype.qtype_analysis import QTypeInstance


def main():
    all_insts: List[QTypeInstance] = load_pickle_from(at_output_dir("qtype", "mmd_qtype_G_pred.parsed.pickle"))
    random.shuffle(all_insts)
    all_insts = all_insts[:10 * 1000]
    n_insts = len(all_insts)
    # clusters, counter = get_clusters(all_insts)
    # save_to_pickle((clusters, counter), "mmd_qtype_G_pred_cluster")
    (clusters, counter) = load_from_pickle("mmd_qtype_G_pred_cluster")

    cluster_rep = cluster_rep_by_tf_idf(clusters, all_insts)
    cluster_rep_sent = cluster_q_term_counts(clusters, all_insts)
    for key, value in counter.most_common():
        print("Cluster {0} ({1:.3f})".format(key, value/n_insts))
        print(cluster_rep[key])
        print(cluster_rep_sent[key])


def get_clusters(all_insts):
    counter = Counter()
    n_cluster = 100
    clusters = [list() for _ in range(n_cluster)]
    for i, inst in enumerate(all_insts):
        for j in range(n_cluster):
            if inst.qtype_weights[j] > 0.5:
                counter[j] += inst.qtype_weights[j]
                clusters[j].append(i)
    return clusters, counter


def get_soft_cluster_info(all_insts: List[QTypeInstance]):
    counter = Counter()
    n_cluster = 100
    strong_elements = [list() for _ in range(n_cluster)]
    cluster_tf_d = [Counter() for _ in range(n_cluster)]

    for i, inst in enumerate(all_insts):
        for j in range(n_cluster):
            for t in inst.get_function_terms():
                cluster_tf_d[j][t] += inst.qtype_weights[j]

            if inst.qtype_weights[j] > 0.01:
                counter[j] += inst.qtype_weights[j]
            if inst.qtype_weights[j] > 0.5:
                strong_elements[j].append(i)
    return strong_elements, counter



if __name__ == "__main__":
    main()
