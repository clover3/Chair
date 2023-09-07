from table_lib import tsv_iter
from misc_lib import group_by, get_first
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import term_align_candidate2_score_path


def main():
    save_path = term_align_candidate2_score_path()

    entries = list(tsv_iter(save_path))
    for q_term, g_entries in group_by(entries, get_first).items():
        pos_term = []
        neg_term = []
        for q_term_, d_term, score in g_entries:
            if float(score) > 0:
                pos_term.append(d_term)
            else:
                neg_term.append(d_term)

        print("Q Term: ", q_term)
        print("Pos: ", " ".join(pos_term))
        print("Neg: ", " ".join(neg_term))


if __name__ == "__main__":
    main()
