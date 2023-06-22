import json
import os
from collections import Counter

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_label_json_path
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem
from cpath import output_path
from list_lib import index_by_fn
from tab_print import print_table, tab_print_dict


def get_problem_id(d):
    return d['group_no'], d['inner_idx']


def main():
    problems = load_alamri_problem()
    problem_d = index_by_fn(lambda p: (p.group_no, p.inner_idx), problems)

    labels = json.load(open(get_sbl_label_json_path(), "r"))
    counter = Counter()
    for l in labels:
        p = problem_d[get_problem_id(l)]
        tokens1 = p.text1.split()
        tokens2 = p.text2.split()

        counter['claim pair'] += 1

        n_tokens = len(tokens1) + len(tokens2)
        counter['num tokens'] += n_tokens
        # n_all_tokens += n_tokens
        for key in l['label']:
            n_items = len(l['label'][key])
            counter["pos tokens"] += n_items
            if "mismatch" in key:
                counter["mismatch tokens"] += n_items
            elif "conflict" in key:
                counter["conflict tokens"] += n_items

    tab_print_dict(counter)


if __name__ == "__main__":
    main()
