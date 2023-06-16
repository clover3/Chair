import json
import os

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_label_json_path
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem
from cpath import output_path
from list_lib import index_by_fn
from tab_print import print_table


def get_problem_id(d):
    return d['group_no'], d['inner_idx']


def main():
    problems = load_alamri_problem()
    problem_d = index_by_fn(lambda p: (p.group_no, p.inner_idx), problems)


    labels = json.load(open(get_sbl_label_json_path(), "r"))
    n_all_tokens = 0
    n_all_pos_annot = 0
    for l in labels:
        p = problem_d[get_problem_id(l)]
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()

        n_tokens = len(tokens1) + len(tokens2)
        n_all_tokens += n_tokens
        for key in l['label']:
            n_items = len(l['label'][key])
            n_all_pos_annot += n_items

    table = [
        ['all tokens', n_all_tokens],
        ['pos token', n_all_pos_annot],
        ['num pairs', len(labels)]
    ]
    print_table(table)

if __name__ == "__main__":
    main()