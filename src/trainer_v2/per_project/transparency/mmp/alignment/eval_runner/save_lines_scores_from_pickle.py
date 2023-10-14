import os.path
import sys
from cpath import output_path
from misc_lib import path_join

from cache import load_pickle_from


def save_line_scores(scores, save_path):
    with open(save_path, "w") as f:
        for row in scores:
            s = row[0]
            f.write("{}\n".format(s))


def main():
    prediction_path = sys.argv[1]
    try:
        target_prediction = sys.argv[2]
    except IndexError:
        target_prediction = "align_pred"

    run_name = os.path.basename(prediction_path)
    output = load_pickle_from(prediction_path)
    pred_score = output['align_probe'][target_prediction]
    lines_save_path = path_join(output_path, "lines_scores", run_name + ".txt")
    save_line_scores(pred_score, lines_save_path)


if __name__ == "__main__":
    main()