from misc_lib import get_dir_files
from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand4_path_helper


def main():
    path_helper = get_cand4_path_helper()
    itr = tsv_iter(path_helper.per_pair_candidates.fidelity_table_path)
    pos_entry = []
    for q_term, d_term, score in itr:
        q_tokens = q_term.split()
        if d_term in q_tokens:
            continue
        if float(score) > 0.1:
            pos_entry.append((q_term, d_term, score))

    print_table(pos_entry)


if __name__ == "__main__":
    main()
