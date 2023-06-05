import sys
from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cache import load_list_from_jsonl
from list_lib import lflatten
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import compare_fidelity, \
    pearson_r_wrap, TermEffectPerQuery
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_path_base
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores


def get_score(q_term, d_term):
    job_no_itr = get_mmp_split_w_deep_scores()
    all_items: List[List[TermEffectPerQuery]] = []
    for job_no in job_no_itr:
        try:
            save_path = get_te_save_path_base(q_term, d_term, job_no)
            item: List[TermEffectPerQuery] = load_list_from_jsonl(save_path, TermEffectPerQuery.from_json)
            all_items.append(item)
        except FileNotFoundError:
            pass
    if len(job_no_itr) != len(all_items):
        raise ValueError()
    te_list: List[TermEffectPerQuery] = lflatten(all_items)
    fidelity_pair_list: List[Tuple[float, float]] = [compare_fidelity(te, pearson_r_wrap) for te in te_list]
    counter = Counter()
    delta_sum = 0
    for t1, t2 in fidelity_pair_list:
        if t1 < t2:
            counter["<"] += 1
        elif t1 == t2:
            counter["=="] += 1
        else:
            counter[">"] += 1

        delta = t2 - t1
        delta_sum += delta
    return delta_sum


def show_te(q_term, d_term):
    c_log.info("Loading job_no iter")
    job_no_itr = get_mmp_split_w_deep_scores()
    c_log.info("Loading te jsonls")
    all_items: List[List[TermEffectPerQuery]] = []
    for job_no in job_no_itr:
        try:
            save_path = get_te_save_path_base(q_term, d_term, job_no)
            item: List[TermEffectPerQuery] = load_list_from_jsonl(save_path, TermEffectPerQuery.from_json)
            all_items.append(item)
        except FileNotFoundError:
            pass

    if len(job_no_itr) != len(all_items):
        print(f"out of {len(job_no_itr)} jobs {len(all_items)} are actually loaded")
    c_log.info("Measure fidelity")
    te_list: List[TermEffectPerQuery] = lflatten(all_items)
    fidelity_pair_list: List[Tuple[float, float]] = [compare_fidelity(te, pearson_r_wrap) for te in te_list]
    counter = Counter()
    delta_sum = 0
    for t1, t2 in fidelity_pair_list:
        if t1 < t2:
            counter["<"] += 1
        elif t1 == t2:
            counter["=="] += 1
        else:
            counter[">"] += 1

        delta = t2 - t1
        delta_sum += delta
    print(counter)
    print(delta_sum)
    c_log.info("Done")


def main():
    q_term = sys.argv[1]
    d_term = sys.argv[2]
    show_te(q_term, d_term)


if __name__ == "__main__":
    main()