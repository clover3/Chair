import numpy as np
import sys
from collections import Counter, defaultdict

from cpath import output_path
from list_lib import right
from misc_lib import path_join, get_second, average
from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list




def analyze_inner(score_log, term_list):
    seen_doc = set()
    rank_per_term = defaultdict(list)

    for row in score_log:
        doc_id = row[1]
        if doc_id in seen_doc:
            continue

        seen_doc.add(doc_id)
        q_id = row[0]
        scores = list(map(float, row[2:]))
        rank_idx = np.argsort(scores)[::-1]

        for rank, idx in enumerate(rank_idx):
            term = term_list[idx]
            rank_per_term[term].append(rank)


    table = []
    for term, ranks in rank_per_term.items():
        avg_rank = average(ranks)
        row = [term, avg_rank]
        table.append(row)
    return table

    # print_table(table)


def main():
    method_list = [
        "ce_mini_lm",
        "splade",
        "tas_b",
        "contriever",
        "contriever-msmarco",
    ]
    term_list = load_car_maker_list()
    for method in method_list:
        score_log_path = f"output/mmp/bias/car_exp/exp3/{method}.tsv"
        score_log = list(tsv_iter(score_log_path))
        table = analyze_inner(score_log, term_list)
        rank_save_path = f"output/mmp/bias/car_exp/exp3/{method}_rank.tsv"
        save_tsv(table, rank_save_path)


if __name__ == "__main__":
    main()
