import csv

from cpath import output_path
from misc_lib import path_join, TimeEstimator
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.runner.summarize_te import get_score


def main():
    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")
    todo_list = [line.strip() for line in open(cand_path, "r")]

    n_error_streak = 0
    rows = []
    ticker = TimeEstimator(3000)
    for i in range(10000):
        ticker.tick()
        q_term, d_term = todo_list[i].split()
        try:
            score = get_score(q_term, d_term)
            row = [q_term, d_term, score]
            rows.append(row)
            print(row)
            n_error_streak = 0
        except ValueError:
            n_error_streak += 1

            if n_error_streak > 10:
                break

    save_path = path_join(
        output_path, "msmarco", "passage", "align_scores", "candidate1.tsv")
    f_out = csv.writer(open(save_path, "w", encoding="utf-8"), dialect='excel-tab')
    f_out.writerows(rows)



if __name__ == "__main__":
    main()