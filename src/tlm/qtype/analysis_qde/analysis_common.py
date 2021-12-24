import random
from collections import Counter
from typing import List

import numpy as np

from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from misc_lib import tprint, TEL, TimeEstimator
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list, convert_ids_to_tokens
from tlm.qtype.qtype_analysis import QTypeInstance2
from tlm.qtype.qtype_instance import QTypeInstance


def group_by_dim_threshold(all_insts: List[QTypeInstance2]):
    n_cluster = 1000
    clusters = [list() for _ in range(n_cluster)]
    for j, inst in enumerate(all_insts):
        qtype_weights = inst.qtype_weights_qe
        ranked_dims = np.argsort(qtype_weights)[::-1]
        for d in ranked_dims[:10]:
            if qtype_weights[d] > 0.5:
                c: List = clusters[d]
                c.append(j)

    for c_idx, c in enumerate(clusters):
        if not len(c):
            continue

        print("Cluster {}".format(c_idx))
        print("# Distinct queries :", len(c))
        print("<<<")
        s_counter = Counter()
        for j in c:
            s = all_insts[j].summary()
            s_counter[s] += 1
        for s, cnt in s_counter.most_common():
            print(cnt, s)


def calculate_cut_inner(g_vectors: List[np.array]):
    random.shuffle(g_vectors)
    n = 10000
    part = g_vectors[:n]
    part = np.stack(part, axis=0)
    print('part', part.shape)
    cut = np.ones_like(part)
    num_over_cut = np.sum(np.less(cut, part).astype(int), axis=0)
    print('num_over_cut', num_over_cut.shape)
    print("Print # insts with dimension value exceed threshold")
    for i in range(11):
        percent = i / 100
        num_range_end = n * percent
        n_over = np.sum(np.less(num_over_cut, num_range_end).astype(int))
        print(" < {0:.2f} : {1}".format(percent, n_over))

    # vector2d = np.stack(part, axis=0)
    # vector_sorted = np.sort(vector2d, axis=0)


def caculate_cut(all_insts: List[QTypeInstance]):
    all_insts = all_insts
    g_vectors = []
    tprint("Enumerating instances")
    for inst in TEL(all_insts):
        g_vectors.append(inst.qtype_weights_de)
    calculate_cut_inner(g_vectors)


def get_avg_vector(vector_list, any_vector):
    if vector_list:
        cur_avg = np.zeros_like(vector_list[0])
        n_prev_items = 0
        st = 0
        step = 10000
        while st < len(vector_list):
            ed = st + step
            window = vector_list[st:ed]
            window_mean = np.mean(np.stack(window, axis=0), axis=0)
            n_cur_items = len(window)
            cur_avg = (cur_avg * n_prev_items + window_mean * n_cur_items) / (n_prev_items + n_cur_items)
            st = ed
            n_prev_items += n_cur_items
        return cur_avg
    else:
        return np.zeros_like(any_vector)


def doc_side_analysis(all_insts: List[QTypeInstance]):
    n_voca = 30522
    vector_per_term = [list() for _ in range(n_voca)]
    all_insts = all_insts
    g_vectors = []
    df = Counter()
    tokenizer = get_tokenizer()
    tprint("Enumerating instances")
    for inst in TEL(all_insts):
        seg1, seg2 = split_p_h_with_input_ids(inst.de_input_ids, inst.de_input_ids)
        doc = seg2
        for token in set(doc):
            df[token] += 1
            vector_per_term[token].append(inst.qtype_weights_de)
        g_vectors.append(inst.qtype_weights_de)

    is_empty = list(map(bool, vector_per_term))

    tprint("Calculating avg")
    max_df = len(all_insts) * 0.5
    min_df = len(all_insts) * 0.01
    per_token_avg: List[np.array] = []
    vector_per_term_filtered = []
    n_valid_voca = 0
    for e in vector_per_term:
        if min_df < len(e) < max_df:
            vector_per_term_filtered.append(e)
            n_valid_voca += 1
        else:
            vector_per_term_filtered.append([])

    ticker = TimeEstimator(n_valid_voca)
    any_vector = g_vectors[0]
    for e in vector_per_term_filtered:
        if e:
            per_token_avg.append(get_avg_vector(e, any_vector))
            ticker.tick()
        else:
            per_token_avg.append(get_avg_vector([], any_vector))

    per_token_avg_np = np.stack(per_token_avg, axis=0)
    g_avg = get_avg_vector(g_vectors) # [n_dim, ]
    per_token_avg_diff = per_token_avg_np - np.expand_dims(g_avg, 0) # [n_voca, n_dim]

    rank = np.argsort(per_token_avg_diff, axis=0)
    per_dimension_max = np.max(per_token_avg_diff, axis=0)
    threshold = per_dimension_max * 0.5
    binary = np.less(np.expand_dims(threshold, 0), per_token_avg_diff)
    num_sbword_per_dim = np.sum(binary.astype(np.int), axis=0)
    num_to_show = 10
    def dimension_desc(dim):
        items = []
        for k in range(n_voca):
            idx = rank[k, dim]
            if df[idx] > min_df:
                items.append(idx)

            if len(items) > num_to_show:
                break
        return tokenizer.convert_ids_to_tokens(items)

    n_dimension = per_token_avg_diff.shape[1]
    tprint("Now printing")
    for dim_idx in range(n_dimension):
        sbword_items = dimension_desc(dim_idx)
        if is_empty[dim_idx]:
            pass
            # print("{} is empty".format(dim_idx))
        else:
            print("{}\t{}\t{}".format(dim_idx, num_sbword_per_dim[dim_idx], sbword_items))


def get_passsage_fn():
    tokenizer = get_tokenizer()
    voca_list = get_voca_list(tokenizer)
    def get_passage(input_ids):
        seg1, seg2 = split_p_h_with_input_ids(input_ids, input_ids)
        seg2_tokens = convert_ids_to_tokens(voca_list, seg2)
        passage: str = pretty_tokens(seg2_tokens, True)
        return passage

    return get_passage