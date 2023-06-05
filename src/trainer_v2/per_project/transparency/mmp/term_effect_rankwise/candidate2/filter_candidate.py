import random
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join

from misc_lib import group_by
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate1.filter_candidate import \
    get_filtered_candidates_from_when_corpus, compute_metric, CandidateEntry



def select_top_n_percent(entries, rate, n_total):
    n_top = int(n_total * rate)
    return entries[:n_top]


def select_bottom_n_percent(entries, rate, n_total):
    n_bottom = int(n_total * rate)
    return entries[-n_bottom:]


def select_random_middle_n_percent(entries, top_rate, bottom_rate, n_total):
    n_top = int(n_total * top_rate)
    n_bottom = int(n_total * bottom_rate)
    middle_items_all = entries[n_top: -n_bottom]
    random.shuffle(middle_items_all)
    n_middle = n_total - n_top - n_bottom
    middle_items = middle_items_all[:n_middle]
    return middle_items


def main():
    items: List[CandidateEntry] = get_filtered_candidates_from_when_corpus()
    grouped = group_by(items, lambda e: e.q_term)

    pos_rate = 0.4
    mid_rate = 0.3
    neg_rate = 0.3
    target_n = 10000
    n_max_per_group = 100
    n_q_term = 0
    n_all_selected = 0
    all_selected = []
    for q_term, entries in grouped.items():
        entries.sort(key=compute_metric, reverse=True)
        n_total = min(n_max_per_group, len(entries))
        top_items = select_top_n_percent(entries, pos_rate, n_total)
        middle_items = select_bottom_n_percent(entries, neg_rate, n_total)
        bottom_items = select_random_middle_n_percent(entries, pos_rate, neg_rate, n_total)
        selected: List[CandidateEntry] = top_items + middle_items + bottom_items
        all_selected.append(selected)
        n_all_selected += len(selected)
        n_q_term += 1

        print(f"Selected {len(selected)} from {q_term}")
        if n_all_selected >= target_n:
            break

    print(f"selected {n_all_selected}  over {n_q_term} q terms")
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2.tsv")

    save_f = open(save_path, "w")
    for entries in all_selected:
        for e in entries:
            row = e.q_term, e.d_term
            out_line = "\t".join(map(str, row))
            save_f.write(out_line + "\n")


if __name__ == "__main__":
    main()