import sys
from typing import List, Dict, Tuple, Iterator

import scipy.special

from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.parse_cpnr_results import get_recover_subtokens, read_passage_scores
from arg.perspectives.runner_ppnc.show_score_stat import load_cppnc_related_data
from arg.perspectives.types import CPIDPair
from arg.qck.doc_value_calculator import doc_value
from cache import load_from_pickle
from list_lib import lmap
from misc_lib import average, bool_to_yn
from tab_print import print_table


def doc_score_predictions() -> Iterator[Tuple[int, List[float]]]:
    passage_score_path = sys.argv[2]
    yield from load_doc_score_prediction(passage_score_path)


def load_doc_score_prediction(passage_score_path) -> Iterator[Tuple[int, List[float]]]:
    recover_subtokens = get_recover_subtokens()
    data_id_to_info: Dict = load_from_pickle("pc_dev_passage_payload_info")
    grouped_scores: Dict[int, List[Dict]] = read_passage_scores(passage_score_path, data_id_to_info, recover_subtokens)

    def get_score_from_logit(logits):
        return scipy.special.softmax(logits)[1]

    for cid, passages in grouped_scores.items():
        scores: List[float] = lmap(lambda d: get_score_from_logit(d['logits']), passages)
        yield cid, scores


def main():
    baseline_cid_grouped, cid_grouped, claim_d = load_cppnc_related_data()
    gold = get_claim_perspective_id_dict()
    doc_scores = dict(doc_score_predictions())

    for cid, pid_entries in cid_grouped.items():
        baseline_pid_entries = baseline_cid_grouped[cid]

        baseline_score_d = {}
        for cpid, a_thing_array in baseline_pid_entries:
            _, pid = cpid
            assert len(a_thing_array) == 1
            score = a_thing_array[0]['score']
            baseline_score_d[pid] = score

        gold_pids = gold[cid]

        def get_score_per_pid_entry(p_entries: Tuple[CPIDPair, List[Dict]]):
            cpid, entries = p_entries
            return average(lmap(lambda e: e['score'], entries))

        pid_entries.sort(key=get_score_per_pid_entry, reverse=True)

        s = "{} : {}".format(cid, claim_d[cid])
        print(s)
        num_docs = len(pid_entries[0][1])
        doc_value_arr = list([list() for _ in range(num_docs)])
        labels = []
        for cpid, things in pid_entries:
            _, pid = cpid
            label = any([pid in pids for pids in gold_pids])
            labels.append(label)
            base_score = baseline_score_d[pid]
            for doc_idx, per_doc in enumerate(things):
                score = per_doc['score']
                value = doc_value(score, base_score, int(label))
                doc_value_arr[doc_idx].append(value)

        head = ["avg", "pred"] + lmap(bool_to_yn, labels)
        rows = [head]
        doc_score = doc_scores[cid]
        assert len(doc_value_arr) == len(doc_score)

        for pred_score, doc_values in zip(doc_score, doc_value_arr):
            avg = average(doc_values)
            row_float = [avg, pred_score] + doc_values
            row = lmap(lambda x: "{0}".format(x), row_float)
            rows.append(row)
        print_table(rows)


if __name__ == "__main__":
    main()
