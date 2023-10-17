import os.path
import sys

from trainer_v2.per_project.transparency.mmp.alignment.eval_runner.galign_eval_line_scores import run_eval_w_tsv
from trainer_v2.per_project.transparency.mmp.alignment.eval_runner.galign_tune import run_tune


def main():
    val_tsv_path = sys.argv[1]
    val_score_path = sys.argv[2]
    test_tsv_path = sys.argv[3]
    test_score_path = sys.argv[4]
    target_metric = "f1"

    run_name = os.path.basename(test_score_path)
    cut = run_tune(val_score_path, val_tsv_path, target_metric)
    run_eval_w_tsv(test_score_path, test_tsv_path, target_metric, cut, run_name)


if __name__ == "__main__":
    main()