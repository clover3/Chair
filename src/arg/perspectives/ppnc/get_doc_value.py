import os
from typing import List, Dict, Tuple

import scipy.special

from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.pdcd_eval import get_confidence_or_rel_score
from arg.perspectives.types import CPIDPair
from arg.qck.prediction_reader import load_prediction_with_info, group_by_qid_cid
from cpath import output_path
from estimator_helper.output_reader import load_combine_info_jsons
from list_lib import lmap
from misc_lib import exist_or_mkdir, group_by, average
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def load_cppnc_score(save_name, fetch_field_list=None) -> Dict[str, Dict[str, List[Dict]]]:
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, save_name + ".info")
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    predictions = load_prediction_with_info(info_file_path, pred_file_path, fetch_field_list)
    qid_grouped = group_by_qid_cid(predictions)
    return qid_grouped


def load_baseline(save_name):
    out_dir = os.path.join(output_path, "cppnc")
    # 2. Load baseline scores
    baseline_info_file_path = os.path.join(out_dir, save_name + ".info")
    baseline_pred_file_path = os.path.join(out_dir, save_name + ".score")
    baseline_cid_grouped = load_and_group_predictions(baseline_info_file_path, baseline_pred_file_path)
    return baseline_cid_grouped


def load_and_group_predictions(info_file_path, pred_file_path):
    scores: Dict[CPIDPair, List[Dict]] = group_by_cpid(info_file_path, pred_file_path)
    cid_grouped: Dict[int, List[Tuple[CPIDPair, List[Dict]]]] = group_by(scores.items(), lambda x: x[0][0])
    return cid_grouped


def group_by_cpid(info_dir, prediction_file) -> Dict[CPIDPair, List[Dict]]:
    info = load_combine_info_jsons(info_dir)

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]
    scores: List[Dict] = collect_data_w_cpid(prediction_file, info, logit_to_score_softmax)
    grouped: Dict[CPIDPair, List[Dict]] = group_by(scores, lambda x: x['cpid'])
    return grouped


def collect_data_w_cpid(prediction_file, info: Dict, logit_to_score) \
        -> List[Dict]:
    data = EstimatorPredictionViewer(prediction_file)
    print("Num data ", data.data_len)
    out = []
    for entry in data:
        logits = entry.get_vector("logits")
        score = logit_to_score(logits)
        data_id = entry.get_vector("data_id")[0]
        confidence = get_confidence_or_rel_score(entry)
        try:
            cur_info = info[str(data_id)]
            cid = cur_info['cid']
            pid = cur_info['pid']

            cpid = CPIDPair((cid, pid))
            cur_info['cpid'] = cpid
            cur_info['score'] = score
            cur_info['confidence'] = confidence
            out.append(cur_info)
        except KeyError as e:
            print("Key error")
            print("data_id", data_id)
            pass
    return out


# output : (cid, doc_id, passage_idx) -> score

def load_passage_score_d(cppnc_save_name, baseline_save_name) -> Dict[Tuple[str, str, int], float]:
    cid_grouped: Dict[str, Dict[str, List[Dict]]] = load_cppnc_score(cppnc_save_name)
    gold = get_claim_perspective_id_dict()
    baseline_cid_grouped = load_baseline(baseline_save_name)

    score_d: Dict[Tuple[str, str, int], float] = {}

    def get_score_from_entry(entry):
        logit = entry['logits']
        return scipy.special.softmax(logit)[1]

    for cid, pid_entries_d in cid_grouped.items():
        pid_entries_d: Dict[str, List[Dict]] = pid_entries_d
        baseline_pid_entries = baseline_cid_grouped[int(cid)]
        baseline_score_d = fetch_score(baseline_pid_entries)

        gold_pids = gold[int(cid)]

        value_arr_pid_row = []
        for pid, entries_for_pid in pid_entries_d.items():
            label = any([pid in pids for pids in gold_pids])
            base_score = baseline_score_d[int(pid)]

            def get_value_from_entry(entry) -> float:
                score = get_score_from_entry(entry)
                value = doc_value(score, base_score, int(label))
                return value
            cur_value_row: List[float] = lmap(get_value_from_entry, entries_for_pid)
            value_arr_pid_row.append(cur_value_row)

        value_arr_doc_row: List[List[float]] = list(map(list, zip(*value_arr_pid_row)))
        avg_value = lmap(average, value_arr_doc_row)

        doc_info = []
        for pid, entries_for_pid in pid_entries_d.items():
            for entry in entries_for_pid:
                e = entry['kdp'].doc_id, entry['kdp'].passage_idx
                doc_info.append(e)
            break

        assert len(avg_value) == len(doc_info)

        for value, (doc_id, passage_idx) in zip(avg_value, doc_info):
            key = cid, doc_id, passage_idx
            score_d[key] = value

    return score_d


def fetch_score(baseline_pid_entries):
    baseline_score_d = {}
    for cpid, a_thing_array in baseline_pid_entries:
        _, pid = cpid
        assert len(a_thing_array) == 1
        score = a_thing_array[0]['score']
        baseline_score_d[pid] = score
    return baseline_score_d


def doc_value(score_baseline, score, gold):
    error_baseline = gold - score_baseline  # -1 ~ 1
    error = gold - score
    improvement = abs(error_baseline) - abs(error)
    return improvement
