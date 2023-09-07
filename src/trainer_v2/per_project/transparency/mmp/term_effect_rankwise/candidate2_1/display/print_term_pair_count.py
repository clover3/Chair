from collections import Counter

from cpath import output_path
from table_lib import tsv_iter
from tab_print import tab_print_dict
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper


def main():
    ph = get_cand2_1_path_helper()
    file_path = ph.per_pair_candidates.fidelity_table_path

    counter = Counter()
    for q_term, d_term, score_s in tsv_iter(file_path):
        score = float(score_s)
        counter['total'] += 1
        for cut in [1, 0.1, 0]:
            if score >= cut:
                counter[f'over {cut}'] += 1

    tab_print_dict(counter)


if __name__ == "__main__":
    main()
