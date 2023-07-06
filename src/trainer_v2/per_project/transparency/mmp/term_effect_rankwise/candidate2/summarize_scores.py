import csv

from cpath import output_path
from misc_lib import path_join, TEL
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_fidelity_save_path2, \
    term_align_candidate2_score_path, get_fidelity_save_name, mmp_root


def collect_scores_and_save(term_pair_list, fidelity_save_dir, save_path):
    f_out = csv.writer(open(save_path, "w", encoding="utf-8"), dialect='excel-tab')
    for todo in TEL(term_pair_list):
        q_term, d_term = todo
        try:
            save_name = get_fidelity_save_name(d_term, q_term)
            save_path = path_join(fidelity_save_dir, save_name)
            score = float(open(save_path, "r").read())
            row = [q_term, d_term, score]
            f_out.writerow(row)
        except ValueError:
            pass
        except FileNotFoundError:
            pass


def main():
    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2.tsv")
    term_pair_list = [line.strip().split() for line in open(cand_path, "r")]
    save_path = term_align_candidate2_score_path()
    fidelity_save_dir = path_join(mmp_root(), "term_effect_space2", "fidelity")
    collect_scores_and_save(term_pair_list, fidelity_save_dir, save_path)


if __name__ == "__main__":
    main()