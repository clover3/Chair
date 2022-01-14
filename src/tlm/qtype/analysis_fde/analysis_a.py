import os
from collections import Counter
from typing import List, Dict, Iterable

import numpy as np
from scipy.special import softmax

from bert_api.client_lib import BERTClient
from cpath import pjoin, data_path, output_path
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import EncoderUnitPlain, get_tokenizer
from list_lib import left
from misc_lib import NamedAverager, TimeEstimator, group_by, average
from port_info import MMD_Z_PORT
from tlm.qtype.analysis_qde.analysis_common import get_passsage_fn
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.qtype_instance import QTypeInstance
from trainer.np_modules import sigmoid


def enum_count_query(qtype_entries: List[QTypeInstance],
                     query_info_dict: Dict[str, QueryInfo],
                     ):
    query_counter = Counter()
    func_span_counter = Counter()
    qid_counter = Counter()
    for e_idx, e in enumerate(qtype_entries):
        cur_q_info = query_info_dict[e.qid]
        qid_counter[e.qid] += 1
        query_counter[cur_q_info.query] += 1
        functional_span = " ".join(cur_q_info.functional_tokens)
        func_span_counter[functional_span] += 1

    print('number of distinct qid', len(qid_counter))
    print('number of distinct queries', len(query_counter))
    print('number of func_span', len(func_span_counter))


def run_qtype_analysis_a(qtype_entries: Iterable[QTypeInstance],
                         query_info_dict: Dict[str, QueryInfo],
                         q_embedding_d: Dict[str, np.array],
                         q_bias_d: Dict[str, np.array],
                         cluster,
                         apply_sigmoid=False,
                         ):
    get_passage = get_passsage_fn()
    print("Building qtype desc")
    n_func_word = len(q_embedding_d)
    print("{} func_spans are found".format(n_func_word))

    func_span_list = []
    qtype_embedding_flat = []
    q_bias_flat = []
    for idx, key in enumerate(q_embedding_d.keys()):
        func_span_list.append(key)
        qtype_embedding_flat.append(q_embedding_d[key])
        q_bias_flat.append(q_bias_d[key])

    q_bias_np = np.stack(q_bias_flat, axis=0)
    qtype_embedding_np = np.stack(qtype_embedding_flat, axis=0)
    threshold_1 = 0
    threshold_2 = 0
    for e_idx, e in enumerate(qtype_entries):
        f_high_model_score = e.label > 0.5
        cur_q_info = query_info_dict[e.qid]
        print(cur_q_info.query)
        if e_idx % 10 == 0:
            dummy = input("Press enter to continue")
        if apply_sigmoid:
            prob = sigmoid(e.logits)
        else:
            prob = e.logits
        f_high_logit = prob > 0.5

        if f_high_model_score:
            continue
        q_rep = " ".join(cur_q_info.out_s_list)
        func_word_weights_d = np.matmul(qtype_embedding_np, e.qtype_weights_de) + q_bias_np

        n_promising_func_word = np.count_nonzero(np.less(threshold_1, func_word_weights_d))
        if not n_promising_func_word:
            continue

        if apply_sigmoid:
            combined_score = sigmoid(func_word_weights_d)
        else:
            combined_score = func_word_weights_d

        n_promising_func_word2 = np.count_nonzero(np.less(threshold_2, combined_score))
        if not n_promising_func_word2:
            continue
        print("---------------------------------")
        print("e_idx=", e_idx)
        print(e.qid, q_rep)
        print("{}:{} - {}".format(e.doc_id, e.passage_idx, "Relevant" if e.label else "Non-relevant"))
        print("Score bias q_bias d_bias")
        print(" ".join(map("{0:.2f}".format, [e.logits, e.bias, e.q_bias, e.d_bias])))
        # print_combined_score_info(cluster, combined_score, func_span_list, func_word_weights_d)
        print_combined_score_info2(cluster, combined_score, func_span_list, prob)
        print(get_passage(e.de_input_ids))


def print_combined_score_info(cluster, combined_score, func_span_list, func_word_weights_d):
    rank = np.argsort(combined_score)[::-1]
    seen_cluster = set()
    for i in range(100):
        type_i = rank[i]
        cluster_i = cluster[type_i]
        seen_cluster.add(cluster_i)
        score = combined_score[type_i]
        if score < 0:
            break
        func_span = func_span_list[type_i]
        s = "{0} {1:.2f} = {2:.2f}".format(
            func_span,
            score,
            func_word_weights_d[type_i])
        print(s)
        if len(seen_cluster) > 20:
            break


def print_combined_score_info2(cluster, combined_score, func_span_list, threshold):
    cluster_size = Counter()
    for cluster_id in cluster:
        cluster_size[cluster_id] += 1

    TypeID = int
    ClusterID = int
    over_threshold: List[TypeID] = []
    for type_i, s in enumerate(combined_score):
        if s >= threshold:
            over_threshold.append(type_i)

    print("{} functional spans matched. (Threshold={})".format(len(over_threshold), threshold))
    grouped: Dict[ClusterID, List[TypeID]] = group_by(over_threshold, lambda i: cluster[i])

    for cluster_idx in grouped:
        members: List[TypeID] = grouped[cluster_idx]

        avg = average([combined_score[type_i] for type_i in members])
        s1 = f"Cluster {cluster_idx}. {len(members)} of {cluster_size[cluster_idx]} members matched. Avg={avg}"
        spans = " / ".join([func_span_list[type_i] for type_i in members[:30]])
        print(s1)
        print("\t" + spans)



def get_mmd_client_wrap():
    max_seq_length = 512
    client = BERTClient("http://localhost", MMD_Z_PORT, max_seq_length)
    voca_path = pjoin(data_path, "bert_voca.txt")
    d_encoder = EncoderUnitPlain(max_seq_length, voca_path)
    tokenizer = get_tokenizer()
    def query_multiple(query_list: List[str], doc_tokens_ids: List[int]):
        payload = []
        for query in query_list:
            q_tokens_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))
            d = d_encoder.encode_inner(q_tokens_id, doc_tokens_ids)
            p = d["input_ids"], d["input_mask"], d["segment_ids"]
            payload.append(p)
        ret = client.send_payload(payload)
        return ret
    return query_multiple


def AP(gold):
    n_pred_pos = 0
    tp = 0
    sum_prec = 0
    for is_rel in gold:
        n_pred_pos += 1
        if is_rel:
            tp += 1
            sum_prec += (tp / n_pred_pos)
    return sum_prec / np.count_nonzero(gold)


def run_qtype_analysis_b(qtype_entries: Iterable[QTypeInstance],
                         query_info_dict: Dict[str, QueryInfo],
                         q_embedding_d: Dict[str, np.array],
                         apply_sigmoid=False,
                         ):
    print("Building qtype desc")
    n_func_word = len(q_embedding_d)
    print("{} func_spans are found".format(n_func_word))
    query_multiple = get_mmd_client_wrap()
    func_span_list, qtype_embedding_np = embeddings_to_list(q_embedding_d)

    f_log = open(os.path.join(output_path, "qtype", "log.txt"), "w")
    avg = NamedAverager()
    n_eval_target = 20
    n_eval = 0
    ticker = TimeEstimator(n_eval_target)
    for e_idx, e in enumerate(qtype_entries):
        ticker.tick()
        if n_eval >= n_eval_target:
            break

        if apply_sigmoid:
            prob = sigmoid(e.logits)
        else:
            prob = e.logits
        f_relevant = e.label
        f_high_logit = prob > 0.5

        if not f_relevant:
            continue

        n_eval += 1
        cur_q_info = query_info_dict[e.qid]
        func_word_weights_d = np.matmul(qtype_embedding_np, e.qtype_weights_de)
        if apply_sigmoid:
            combined_score = sigmoid(func_word_weights_d)
        else:
            combined_score = func_word_weights_d

        entity, doc = split_p_h_with_input_ids(e.de_input_ids, e.de_input_ids)

        def do_eval_by_ap(per_type_scores):
            rank = np.argsort(per_type_scores)[::-1]
            predictions = []
            for i in range(100):
                type_i = rank[i]
                score = per_type_scores[type_i]
                func_span = func_span_list[type_i]
                new_gen_query = func_span.replace("[MASK]", cur_q_info.content_span)
                predictions.append((new_gen_query, score))
            new_queries = left(predictions)
            ret = query_multiple(new_queries, doc.tolist())
            probs = softmax(ret, axis=1)[:, 1]
            correctness = np.less_equal(relevant_threshold, probs)
            for (new_query, score_pred), real_score in zip(predictions, probs):
                s = "CORRECT" if real_score > 0.5 else "WRONG"
                out_s = "{0}\t{1}\t{2:.2f}\t{3:.2f}".format(new_query, s, score_pred, real_score)
                f_log.write(out_s + "\n")

            if not np.count_nonzero(correctness):
                ap = 0
            else:
                ap = AP(correctness)
            return ap

        score_by_length = np.array([1/len(s) + 0.5 for s in func_span_list])

        random_score = np.random.random(len(combined_score))
        f_log.write("< proposed >")
        ap = do_eval_by_ap(combined_score)
        f_log.write("< random >")
        random_ap = do_eval_by_ap(random_score)
        f_log.write("< length_sort >")
        length_sort_ap = do_eval_by_ap(score_by_length)

        avg.avg_dict['proposed'].append(ap)
        avg.avg_dict['random'].append(random_ap)
        avg.avg_dict['length_sort'].append(length_sort_ap)
        # print("{0} / {1} / {2:.2f} {3:.2f}".format(cur_q_info.query, cur_q_info.content_span, ap, random_ap))
        #
        # for (new_query, score_pred), real_score in zip(predictions, probs):
        #     s = "CORRECT" if real_score > 0.5 else "WRONG"
        #     print("{0}\t{1}\t{2:.2f}\t{3:.2f}".format(new_query, s, score_pred, real_score))
        #


    for k, v in avg.get_average_dict().items():
        print(k, v)


def embeddings_to_list(q_embedding_d):
    func_span_list = []
    qtype_embedding_flat = []
    for idx, key in enumerate(q_embedding_d.keys()):
        func_span_list.append(key)
        qtype_embedding_flat.append(q_embedding_d[key])
    relevant_threshold = 0.5
    qtype_embedding_np = np.stack(qtype_embedding_flat, axis=0)
    return func_span_list, qtype_embedding_np
