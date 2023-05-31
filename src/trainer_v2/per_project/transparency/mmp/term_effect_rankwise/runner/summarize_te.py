from collections import Counter

from cache import load_list_from_jsonl
from list_lib import lflatten
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import compare_fidelity, \
    pearson_r_wrap
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_path
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores


def identity(t):
    return t


def main():
    q_term = "when"
    d_term = "sunday"
    job_no_itr = get_mmp_split_w_deep_scores()

    all_items = []
    for job_no in job_no_itr:
        try:
            save_path = get_te_save_path(q_term, d_term, job_no)
            item = load_list_from_jsonl(save_path, identity)
            all_items.append(item)
        except FileNotFoundError:
            pass

    if len(job_no_itr) != len(all_items):
        print(f"out of {len(job_no_itr)} jobs {len(all_items)} are actually loaded")

    te_list = lflatten(all_items)
    print(f"Total of {len(te_list)} queries are affected")

    fidelity_pair_list = [compare_fidelity(te, pearson_r_wrap) for te in te_list]

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


if __name__ == "__main__":
    main()