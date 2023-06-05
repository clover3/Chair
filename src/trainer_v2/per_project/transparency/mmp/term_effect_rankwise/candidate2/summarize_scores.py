import csv

from cpath import output_path
from misc_lib import path_join, TimeEstimator, TEL
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_fidelity_save_path2
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.runner.summarize_te import get_score


def main():
    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2.tsv")
    todo_list = [line.strip() for line in open(cand_path, "r")]

    rows = []
    save_path = path_join(
        output_path, "msmarco", "passage", "align_scores", "candidate2.tsv")
    f_out = csv.writer(open(save_path, "w", encoding="utf-8"), dialect='excel-tab')

    for todo in TEL(todo_list):
        q_term, d_term = todo.split()
        try:
            save_path = get_fidelity_save_path2(q_term, d_term)
            score = float(open(save_path, "r").read())
            row = [q_term, d_term, score]
            f_out.writerow(row)
        except ValueError:
            pass
        except FileNotFoundError:
            pass



if __name__ == "__main__":
    main()