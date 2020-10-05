import pickle
from collections import Counter
from typing import List, Dict, Tuple

from arg.perspectives.eval_caches import get_eval_candidate_as_pids
from arg.qck.doc_value_calculator import DocValueParts2
from list_lib import lmap
from misc_lib import group_by, average
from tab_print import print_table


def load():
    dvp_pickle_path = "output/cppnc_val_ex_score.summary"
    dvp: List[DocValueParts2] = pickle.load(open(dvp_pickle_path, "rb"))
    return dvp


def get_qid(e: DocValueParts2):
    return e.query.query_id


def get_candidate(e: DocValueParts2):
    return e.candidate.id


def get_doc_id_idx(e: DocValueParts2):
    return e.kdp.doc_id, e.kdp.passage_idx


def get_doc_id(e: DocValueParts2):
    return e.kdp.doc_id


def four_digit_float(f):
    return "{0:.4f}".format(f)

def avg_scores():
    dvp: List[DocValueParts2] = load()
    candidate_d_raw: List[Tuple[int, List[int]]] = get_eval_candidate_as_pids("train")
    candidate_d = {str(k): lmap(str, v) for k, v in candidate_d_raw}

    # Group by doc id
    dvp_qid_grouped: Dict[str, List[DocValueParts2]] = group_by(dvp, get_qid)

    rows = []
    for qid, entries in dvp_qid_grouped.items():
        # Q : How many kdp are useful?
        # Q : Does relevant matter?
        candidate_id_grouped = group_by(entries, get_candidate)
        c = Counter()
        new_rows = []
        new_rows.append(["candidate id", "init_score", "avg_score"])

        for candidate_id, entries2 in candidate_id_grouped.items():
            label = entries2[0].label
            avg_score = average(lmap(lambda x: x.score, entries2))
            initial_score = entries2[0].init_score
            change = avg_score - initial_score
            value_type = good_or_bad(change, label, 0.01)
            c[value_type] += 1
            row = [candidate_id, label, value_type, four_digit_float(initial_score), four_digit_float(avg_score)]
            new_rows.append(row)

        row = [qid, c['good'], c['bad'], c['no change']]
        rows.append(row)
        rows.extend(new_rows)
    print_table(rows)


def to_pred(score):
    return int(score > 0.5)


def direction(score, base_score):
    if score > base_score:
        return "up"
    elif score < base_score:
        return "down"
    else:
        return "no change"


def group_by_cids():
    dvp: List[DocValueParts2] = load()
    candidate_d_raw: List[Tuple[int, List[int]]] = get_eval_candidate_as_pids("train")
    candidate_d = {str(k): lmap(str, v) for k, v in candidate_d_raw}

    # Group by doc id
    dvp_qid_grouped: Dict[str, List[DocValueParts2]] = group_by(dvp, get_qid)

    def simple(doc_id):
        return doc_id.split("-")[-1]

    rows = []
    for qid, entries in dvp_qid_grouped.items():
        # Q : How many kdp are useful?
        # Q : Does relevant matter?
        candidate_id_grouped = group_by(entries, get_candidate)
        rows.append([qid])
        for candidate_id, entries2 in candidate_id_grouped.items():
            is_initial_candidate = candidate_id in candidate_d[qid]
            avg_score = average(lmap(lambda x: x.score, entries2))

            rows.append(['candidate id:', candidate_id])
            rows.append(['is_initial_candidate', is_initial_candidate])
            rows.append(["doc_id", "score", "gold", "init_pred", "direction", "decision"])
            for e in entries2:
                s = "{}_{}".format(simple(e.kdp.doc_id), e.kdp.passage_idx)
                row = [s, "{0:.2f}".format(e.score), e.label, to_pred(e.init_score),
                       direction(e.score, e.init_score), to_pred(e.score)]

                rows.append(row)

    print_table(rows)


def good_or_bad(change, label, t=0.03):
    if label and change > t:
        return "good"
    elif not label and change < t:
        return "good"
    elif label and change < t:
        return "bad"
    elif not label and change > t:
        return "bad"
    else:
        return "no change"


def top_k_avg(l):
    l.sort(reverse=True)
    k = 10
    return average(l[:k])


def get_decision_change(label, base_score, score):
    init_pred: bool = base_score > 0.5
    new_pred: bool = score > 0.5

    if init_pred != new_pred:
        if new_pred == bool(label):
            return "decision_change_good"
        else:
            return "decision_change_bad"
    else:
        return "no_change"


def group_by_docs():
    dvp: List[DocValueParts2] = load()
    candidate_d_raw: List[Tuple[int, List[int]]] = get_eval_candidate_as_pids("train")
    candidate_d = {str(k): lmap(str, v) for k, v in candidate_d_raw}

    # Group by doc id
    dvp_qid_grouped: Dict[str, List[DocValueParts2]] = group_by(dvp, get_qid)

    def simple(doc_id):
        return doc_id.split("-")[-1]


    c_all = Counter()
    rows = []
    for qid, entries in dvp_qid_grouped.items():
        # Q : How many kdp are useful?
        # Q : Does relevant matter?
        candidate_id_grouped = group_by(entries, get_doc_id)
        rows.append(["qid", qid])

        for doc_id_idx, entries2 in candidate_id_grouped.items():
            #c = Counter([good_or_bad(e.score-e.init_score, e.label) for e in entries2])
            c = Counter([get_decision_change(e.label, e.init_score, e.score) for e in entries2])
            rows.append([doc_id_idx])
            #row = [doc_id_idx, c["good"], c["bad"], c["no change"]]
            row = [doc_id_idx, c["decision_change_good"], c["decision_change_bad"], c["no_change"]]
            rows.append(row)
            for k, v in c.items():
                c_all[k] += v

    row = ["summary", c_all["decision_change_good"], c_all["decision_change_bad"], c_all["no_change"]]
    rows = [row] + rows

    print_table(rows)



def sanity_check():
    dvp: List[DocValueParts2] = load()
    candidate_d_raw: List[Tuple[int, List[int]]] = get_eval_candidate_as_pids("train")
    candidate_d = {str(k): lmap(str, v) for k, v in candidate_d_raw}

    # Group by doc id
    dvp_qid_grouped: Dict[str, List[DocValueParts2]] = group_by(dvp, get_qid)

    ap_baseline = []
    ap_new_score = []
    for qid, entries in dvp_qid_grouped.items():
        ranked_list_new = []
        ranked_list_baseline = []

        candidate_id_grouped = group_by(entries, get_candidate)
        for candidate_id, entries2 in candidate_id_grouped.items():
            is_initial_candidate = candidate_id in candidate_d[qid]
            gold = entries2[0].label
            skip = gold and not is_initial_candidate

            def get_new_score(dvp: DocValueParts2):
                return dvp.score

            def get_baseline_score(dvp: DocValueParts2):
                return dvp.init_score

            if skip:
                continue

            new_score = top_k_avg(lmap(get_new_score, entries2))
            baseline_score = average(lmap(get_baseline_score, entries2))
            ranked_list_new.append((candidate_id, new_score, gold))
            ranked_list_baseline.append((candidate_id, baseline_score, gold))

        def get_ap(ranked_list):
            ranked_list.sort(key=lambda x: x[1], reverse=True)

            p_list = []
            p = 0
            for rank, (cid, score, gold) in enumerate(ranked_list):
                if gold:
                    p += 1
                    p_list.append(p/(rank+1))
            return average(p_list)

        ap_baseline.append(get_ap(ranked_list_baseline))
        ap_new_score.append(get_ap(ranked_list_new))

    print("MAP baseline", average(ap_baseline))
    print("MAP new score", average(ap_new_score))


if __name__ == "__main__":
    group_by_docs()
