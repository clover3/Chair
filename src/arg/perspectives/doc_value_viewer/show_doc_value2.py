from typing import List, Dict, Tuple

import scipy.special

from arg.perspectives.load import get_claims_from_ids, \
    get_claim_perspective_id_dict, \
    load_train_claim_ids
from arg.perspectives.ppnc.get_doc_value import load_cppnc_score, load_baseline
from arg.qck.doc_value_calculator import doc_value
from arg.qck.prediction_reader import qk_convert_map, load_combine_info_jsons
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap, left
from misc_lib import group_by, average, bool_to_yn
from tab_print import print_table


def doc_score_predictions():
    passage_score_path = "output/cppnc/qknc_val"
    info = load_combine_info_jsons("output/cppnc/qknc_val.info", qk_convert_map)
    data = join_prediction_with_info(passage_score_path, info)
    grouped: Dict[str, List[Dict]] = group_by(data, lambda x: x['query'].query_id)

    def get_score_from_logit(logits):
        return scipy.special.softmax(logits)[1]

    for cid, passages in grouped.items():
        scores: List[float] = lmap(lambda d: get_score_from_logit(d['logits']), passages)
        yield cid, scores


def load_train_claim_d():
    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claim_d = {c['cId']: c['text'] for c in claims}
    return claim_d


def get_score_from_entry(entry):
    logit = entry['logits']
    return scipy.special.softmax(logit)[1]


def fetch_score_per_pid(baseline_pid_entries) -> Dict[int, float]:
    baseline_score_d = {}
    for cpid, a_thing_array in baseline_pid_entries:
        _, pid = cpid
        assert len(a_thing_array) == 1
        score = a_thing_array[0]['score']
        baseline_score_d[pid] = score
    return baseline_score_d


def collect_score_per_doc(baseline_score_d, get_score_from_entry, gold_pids, pid_entries)\
        -> Tuple[Dict[int, Tuple[str, int]], List[List[float]], List[bool]]:
    num_docs = len(pid_entries[0][1])
    print("Num_docs", num_docs)
    doc_value_arr: List[List[float]] = list([list() for _ in range(num_docs)])
    seen_sig = set()
    doc_info_d: Dict[int, Tuple[str, int]] = {}
    labels = []
    cid = None
    print("Num candidate : ", len(pid_entries))
    for pid, entries in pid_entries:
        label = any([pid in pids for pids in gold_pids])
        labels.append(label)
        base_score = baseline_score_d[int(pid)]
        print("num docs for pid :{} : {} ".format(pid, len(entries)))
        print("Seen sig:", len(seen_sig))
        for doc_idx, entry in enumerate(entries):
            cid_ = entry['query'].query_id
            if cid is None:
                cid = cid_
            assert cid == cid_
            assert pid == entry['candidate'].id
            sig = pid, entry['kdp'].doc_id, entry['kdp'].passage_idx
            seen_sig.add(sig)
            if doc_idx < num_docs:
                score = get_score_from_entry(entry)
                value = doc_value(score, base_score, int(label))
                doc_value_arr[doc_idx].append(value)
                doc_info_d[doc_idx] = entry['kdp'].doc_id, entry['kdp'].passage_idx
            else:
                print()

    return doc_info_d, doc_value_arr, labels


def main():
    print("Loading doc score")
    doc_scores = dict(doc_score_predictions())
    print("Loading cppnc scores")
    save_name = "qcknc_val"
    cid_grouped: Dict[str, Dict[str, List[Dict]]] = load_cppnc_score(save_name)
    print(".")

    gold = get_claim_perspective_id_dict()
    baseline_cid_grouped: Dict[int, List] = load_baseline("train_baseline")
    claim_d = load_train_claim_d()

    for cid, pid_entries_d in cid_grouped.items():
        pid_entries_d: Dict[str, List[Dict]] = pid_entries_d
        baseline_pid_entries = baseline_cid_grouped[int(cid)]

        baseline_score_d = fetch_score_per_pid(baseline_pid_entries)

        gold_pids = gold[int(cid)]

        def get_score_per_pid_entry(p_entries: Tuple[str, List[Dict]]):
            _, entries = p_entries
            return average(lmap(get_score_from_entry, entries))

        pid_entries: List[Tuple[str, List[Dict]]] = list(pid_entries_d.items())
        pid_entries.sort(key=get_score_per_pid_entry, reverse=True)

        s = "{} : {}".format(cid, claim_d[int(cid)])
        print(s)
        doc_info_d, doc_value_arr, labels = collect_score_per_doc(baseline_score_d, get_score_from_entry, gold_pids,
                                                                  pid_entries)

        pids = left(pid_entries)
        head1 = [""] * 4 + pids
        head2 = ["avg", "doc_id", "passage_idx", "pknc_pred"] + lmap(bool_to_yn, labels)
        rows = [head1, head2]
        doc_score = doc_scores[cid]
        assert len(doc_value_arr) == len(doc_score)

        for doc_idx, (pred_score, doc_values) in enumerate(zip(doc_score, doc_value_arr)):
            doc_id, passage_idx = doc_info_d[doc_idx]
            avg = average(doc_values)
            row_float = [avg, doc_id, passage_idx, pred_score] + doc_values
            row = lmap(lambda x: "{0}".format(x), row_float)
            rows.append(row)
        print_table(rows)


if __name__ == "__main__":
    main()
