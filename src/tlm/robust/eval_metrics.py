import os
import sys

import numpy as np

from cache import load_pickle_from
from data_generator.common import get_tokenizer
from data_generator.data_parser.robust2 import load_robust_qrel, robust_path
from evals.adhoc import p_at_k, dcg_at_k
from list_lib import left
from misc_lib import average
from taskman_client.task_proxy import get_task_manager_proxy
from tlm.estimator_prediction_viewer import flatten_batches


def generate_ranked_list(tf_prediction_data, payload_info, k):
    tf_prediction_data = flatten_batches(tf_prediction_data)
    logits = tf_prediction_data["logits"]

    scores = np.reshape(logits, [-1])


    g_idx = 0
    pred_list = []

    all_ranked_list = []
    for _ in range(50):
        ranked_list = []
        for _ in range(k):
            score = scores[g_idx]
            doc_id = payload_info[g_idx]
            ranked_list.append((doc_id, score))
            g_idx += 1

        ranked_list.sort(key=lambda x: x[1], reverse=True)

        new_ranked_list = []
        for idx, (doc_id, score) in enumerate(ranked_list):
            new_ranked_list.append((doc_id, idx+1, score))
        all_ranked_list.append(new_ranked_list)

    return all_ranked_list


def write_ranked_list(q_id_list, all_ranked_list, out_path):
    assert len(q_id_list) == len(all_ranked_list)

    f = open(out_path, "w")
    for q_id, ranked_list in zip(q_id_list, all_ranked_list):
        for doc_id, rank, score in ranked_list:
            line = "{} Q0 {} {} {} galago\n".format(q_id, doc_id, rank, score)
            f.write(line)
    f.close()

def parse_prediction_and_eval(prediction_path, payload_type, data_id, k=100):
    payload_info = get_payload_info(payload_type, data_id)
    tf_prediction_data = load_pickle_from(prediction_path)
    all_ranked_list = generate_ranked_list(tf_prediction_data, payload_info, k)

    text_output_path = prediction_path + ".txt"
    st = int(data_id)
    write_ranked_list(range(st, st+50), all_ranked_list, text_output_path)
    pred_list = []
    for ranked_list in all_ranked_list:
        pred = [x[0] for x in ranked_list]
        pred_list.append(pred)

    return eval(pred_list, data_id)


def show(tf_prediction_data, payload_info, data_id, k=100):
    tf_prediction_data = flatten_batches(tf_prediction_data)
    logits = np.reshape(tf_prediction_data["logits"], [-1])
    scores = np.reshape(logits, [-1])
    tokenizer = get_tokenizer()

    g_idx = 0
    pred_list = []

    for _ in range(50):
        ranked_list = []
        for _ in range(k):
            score = scores[g_idx]
            doc_id = payload_info[g_idx]
            g_idx += 1
            print(doc_id, score)
            ranked_list.append((doc_id, score))

        ranked_list.sort(key=lambda x: x[1], reverse=True)
        pred = left(ranked_list)
        pred_list.append(pred)
    qrels = load_robust_qrel()
    gold_list = []
    st = int(data_id)
    query_ids = [str(i) for i in range(st, st + 50)]
    fn = 0
    tn = 0
    for idx, query_id in enumerate(query_ids):
        gold = qrels[query_id] if query_id in qrels else {}
        pred = pred_list[idx]

        print(query_id, "-------------")
        for doc_id in pred[:20]:
            if doc_id in gold:
                if gold[doc_id] == 1 or gold[doc_id] == 2:
                    s = "T"
                elif gold[doc_id] == 0:
                    s = "F"
                else:
                    print(gold[doc_id])
                    assert False
            else:
                s = "N"
            if s == "T":
                fn += 1
            else:
                tn += 1
            print(doc_id, s, )
    print("data len" , len(logits))
    print("accuracy : ", tn/(fn+tn))


def ndcg_at_k_local(pred_list, gold_dict_list, k):
    ndcg_list = []
    for pred, gold_dict in zip(pred_list, gold_dict_list):
        if not gold_dict:
            continue
        r = []
        for p in pred:
            v = gold_dict[p] if p in gold_dict else 0
            r.append(v)

        ideal = sorted(gold_dict.values(), reverse=True)
        max_dcg = dcg_at_k(ideal, k)
        ndcg = dcg_at_k(r, k) / max_dcg
        ndcg_list.append(ndcg)
    return average(ndcg_list)


def eval(pred_list, data_id):
    st = int(data_id)
    query_ids = [str(i) for i in range(st, st + 50)]
    qrels = load_robust_qrel()
    gold_set_list = []
    gold_dict_list = []

    for query_id in query_ids:
        gold = qrels[query_id] if query_id in qrels else {}
        gold_set = set()
        for key in gold:
            if gold[key] >= 1:
                gold_set.add(key)
        gold_set_list.append(gold_set)
        gold_dict_list.append(gold)


    NDCG20 = ndcg_at_k_local(pred_list, gold_dict_list, 20)
    P20= p_at_k(pred_list, gold_set_list, 20)
    print("P20:", P20)
    print("NDCG20:", NDCG20)

    return P20, NDCG20


def get_payload_info(payload_type, data_id):
    return load_pickle_from(os.path.join(robust_path, payload_type, data_id + ".info"))


if __name__ == "__main__":
    payload_type = sys.argv[1]
    prediction_path = sys.argv[2]
    data_id = sys.argv[3]
    #show(tf_prediction_data, payload_info, data_id)
    P20, NDCG20 = parse_prediction_and_eval(prediction_path, payload_type, data_id)
    proxy = get_task_manager_proxy()
    proxy.report_number(sys.argv[2], NDCG20, "NDCG")
