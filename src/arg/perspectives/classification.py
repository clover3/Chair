import csv
from typing import List, Tuple, Dict

from arg.perspectives.basic_analysis import predict_by_elastic_search
from arg.perspectives.classification_header import get_file_path
from arg.perspectives.load import get_claim_perspective_id_dict, load_dev_claim_ids, get_claims_from_ids, \
    load_test_claim_ids
from arg.perspectives.split_helper import train_split
from cache import save_to_pickle, load_from_pickle
from list_lib import flatten, left, right


def generate_classification_payload():
    claims, val = train_split()
    top_k = 50
    pred = predict_by_elastic_search(claims, top_k)
    save_to_pickle(pred, "perspective_cls_train_X")
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 50
    pred = predict_by_elastic_search(claims, top_k)
    save_to_pickle(pred, "perspective_cls_dev_X")


def get_scores(r: List[Tuple[int, int]]) -> Dict:
    tp = sum([1 if a == b == 1 else 0 for a, b in r])
    tn = sum([1 if a == b == 0 else 0 for a, b in r])
    accuracy = (tp+tn) / len(r)

    pp = sum(left(r))
    precision = tp / pp if pp != 0 else 0
    recall = tp / sum(right(r))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def load_payload(split):
    return load_from_pickle("perspective_cls_{}_X".format(split))


def eval_classification(classifier, split):
    payloads = load_payload(split)
    gold = get_claim_perspective_id_dict()

    r = []
    for cid, data_list in payloads:
        gold_pids = gold[cid]
        all_pid_set = set(flatten(gold_pids))
        for p_entry in data_list:
            c_text = p_entry['claim_text']
            p_text = p_entry['perspective_text']
            z = classifier(c_text, p_text)
            y = 1 if p_entry['pid'] in all_pid_set else 0
            r.append((z, y))
    return get_scores(r)


def save_to_csv():
    gold = get_claim_perspective_id_dict()

    def routine(claims, out_path):
        payloads = predict_by_elastic_search(claims, 50)
        head = ['sentence1', 'sentence2', 'gold_label', 'cid', 'pid']
        rows = []
        for cid, data_list in payloads:
            gold_pids = gold[cid]
            all_pid_set = set(flatten(gold_pids))
            for p_entry in data_list:
                c_text = p_entry['claim_text']
                p_text = p_entry['perspective_text']
                y = 1 if p_entry['pid'] in all_pid_set else 0
                row = [c_text, p_text, y, cid, p_entry['pid']]
                rows.append(row)
        f_out = csv.writer(open(out_path, "w", encoding="utf-8"), dialect='excel-tab')
        f_out.writerows([head]+rows)

    claims, val = train_split()
    routine(claims, get_file_path('train'))
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    routine(claims, get_file_path('dev'))
    d_ids: List[int] = list(load_test_claim_ids())
    claims = get_claims_from_ids(d_ids)
    routine(claims, get_file_path('test'))


if __name__ == "__main__":
    save_to_csv()