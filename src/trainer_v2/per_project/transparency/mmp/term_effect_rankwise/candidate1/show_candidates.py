import csv
import os
from collections import Counter

from cpath import output_path
from misc_lib import path_join, TimeEstimator, get_first
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_path_base
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.runner.summarize_te import get_score


def main():
    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")
    todo_list = [line.strip() for line in open(cand_path, "r", encoding="utf-8")]
    entries = []
    prev_q_term = None
    cnt = 0
    for i in range(0, 10000):
        q_term, d_term = todo_list[i].split()
        entries.append((q_term, d_term))

    entries.sort(key=get_first)
    counter = Counter()
    for i in range(10000):
        q_term, d_term = entries[i]
        if q_term == prev_q_term:
            cnt += 1
            print(q_term)
        prev_q_term = q_term
        counter[q_term] += 1

    print(f"cnt={cnt}")
    print(counter)




if __name__ == "__main__":
    main()
