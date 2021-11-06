import math
from collections import Counter
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from cpath import at_output_dir
from misc_lib import get_second
from tlm.qtype.analysis.save_parsed import parse_q_weight_output
from tlm.qtype.qtype_analysis import QTypeInstance


def nearest_neighbor_analysis(all_insts: List[QTypeInstance]):
    X = [inst.qtype_weights for inst in all_insts]
    nbrs = NearestNeighbors(5).fit(X)
    distances, indices = nbrs.kneighbors(X)
    for j, inst in enumerate(all_insts):
        print(inst.summary())
        for k in indices[j][1:]:
            print("\t", all_insts[k].summary())


def kmeans_analysis(all_insts):
    X = [inst.qtype_weights for inst in all_insts]
    n_cluster = 10
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)
    clusters: List[List] = [[] for _ in range(n_cluster)]
    for j, l in enumerate(kmeans.labels_):
        c: List = clusters[l]
        c.append(j)
    f_term_for_list = [Counter() for _ in range(n_cluster)]
    for cluster_idx in range(n_cluster):
        c = clusters[cluster_idx]
        c_counter = f_term_for_list[cluster_idx]
        for j in c:
            for t in all_insts[j].get_function_terms():
                c_counter[t] += 1

    def get_cluster_rep(c_idx):
        terms = []
        c_rep = f_term_for_list[c_idx]
        for t, cnt in c_rep.most_common(5):
            assert type(t) == str
            terms.append(t)
        return " ".join(terms)

    for j, inst in enumerate(all_insts):
        cluster_idx = kmeans.labels_[j]
        print("{} / Cluster={} ({})".format(inst.summary(), cluster_idx, get_cluster_rep(cluster_idx)))


def cluster_rep_by_log_odd(clusters: List[List[int]], all_insts: List[QTypeInstance]):
    background_tf = Counter()
    for inst in all_insts:
        for t in inst.get_function_terms():
            background_tf[t] += 1

    bg_tf_sum = sum(background_tf.values())

    summaries = []
    for cluster_idx, c in enumerate(clusters):
        cluster_tf = Counter()
        for j in c:
            inst = all_insts[j]
            for t in inst.get_function_terms():
                cluster_tf[t] += 1

        ctf = sum(cluster_tf.values())
        entries = []
        for term, cnt in cluster_tf.items():
            if cnt > 2:
                odd = math.log(cnt / ctf) - math.log(background_tf[term] / bg_tf_sum)
                entries.append((term, odd))

        entries.sort(key=get_second, reverse=True)

        cluster_summary = " ".join(["{0}({1:.2f})".format(term, odd) for term, odd in entries[:10]])
        summaries.append(cluster_summary)
    return summaries


def cluster_rep_by_tf_idf(clusters: List[List[int]], all_insts: List[QTypeInstance]):
    df = Counter()
    for inst in all_insts:
        for t in inst.get_function_terms():
            df[t] += 1

    df_sum = sum(df.values())

    summaries = []
    for cluster_idx, c in enumerate(clusters):
        cluster_tf = Counter()
        for j in c:
            inst = all_insts[j]
            for t in inst.get_function_terms():
                cluster_tf[t] += 1

        ctf = sum(cluster_tf.values())
        entries = []
        for term, cnt in cluster_tf.items():
            idf = math.log(df_sum / df[term]+1)
            entries.append((term, cnt * idf))

        entries.sort(key=get_second, reverse=True)

        cluster_summary = " ".join(["{0}({1:.2f})".format(term, odd) for term, odd in entries[:10]])
        summaries.append(cluster_summary)
    return summaries


def cluster_q_term_counts(clusters: List[List[int]], all_insts: List[QTypeInstance]):
    df = Counter()
    for inst in all_insts:
        for t in inst.get_function_terms():
            df[t] += 1

    summaries = []
    for cluster_idx, c in enumerate(clusters):
        cluster_sent_count = Counter()
        for j in c:
            inst = all_insts[j]
            key = " ".join(inst.get_function_terms())
            cluster_sent_count[key] += 1

        cluster_summary = "\n".join(["{0}({1})".format(sent, cnt) for sent, cnt in cluster_sent_count.most_common(10)])
        summaries.append(cluster_summary)
    return summaries


def vector_dim_analysis(all_insts: List[QTypeInstance]):
    n_cluster = 100
    clusters = [list() for _ in range(n_cluster)]
    for j, inst in enumerate(all_insts):
        qtype_weights = inst.qtype_weights
        ranked_dims = np.argsort(qtype_weights)[::-1]
        assert sum(qtype_weights) < 1.01
        for d in ranked_dims[:10]:
            if qtype_weights[d] > 0.5:
                c: List = clusters[d]
                c.append(j)

            # s = "{0}({1:.2f})".format(d, qtype_weights[d])
            # s_l.append(s)
        # print(inst.label, inst.summary())
        # print(" ".join(s_l))
    cluster_summaries = cluster_rep_by_log_odd(clusters, all_insts)

    print("Num clusters:", len([c for c in cluster_summaries if c]))
    for c_idx, c in enumerate(clusters):
        if not len(c):
            continue

        print("Cluster {}".format(c_idx))
        print("Top log-odd", cluster_summaries[c_idx])
        print("# Distinct queries :", len(c))
        print("<<<")
        s_counter = Counter()
        for j in c:
            s = all_insts[j].summary()
            s_counter[s] += 1
        for s, cnt in s_counter.most_common():
            print(cnt, s)


def main():
    run_name = "mmd_qtype_G_pred"
    run_name = "mmd_qtype_I"
    run_name = "mmd_qtype_O_200000"
    raw_prediction_path = at_output_dir("qtype", run_name)
    all_insts = parse_q_weight_output(raw_prediction_path)
    vector_dim_analysis(all_insts)


if __name__ == "__main__":
    main()
