import os

from alignment.ists_eval.f1_calc import calc_f1
from alignment.ists_eval.matrix_eval_helper import load_ists_predictions
from alignment.ists_eval.path_helper import get_ists_save_path
from dataset_specific.ists.parse import AlignmentPredictionList
from dataset_specific.ists.path_helper import load_ists_label
from dataset_specific.ists.split_info import ists_enum_split_genre_combs
from tab_print import print_table


def show_scores(genre, run_name_list, split):
    gold: AlignmentPredictionList = load_ists_label(genre, split)
    mode_list = ["", "type", "score"]
    mode_list = ["type"]
    head = ["Run_name"] + mode_list
    table = [head]
    for run_name in run_name_list:
        pred: AlignmentPredictionList = load_ists_predictions(genre, split, run_name)
        row = [run_name]
        for mode in mode_list:
            scores = calc_f1(gold, pred, mode)
            # s = [scores['f1'], scores["precision"], scores["recall"]]
            s = scores["f1"]
            row.append(s)
        table.append(row)
    print_table(table)


def get_typed_f1(genre, split, run_name):
    pred: AlignmentPredictionList = load_ists_predictions(genre, split, run_name)
    gold: AlignmentPredictionList = load_ists_label(genre, split)
    scores = calc_f1(gold, pred, "type")
    return scores["f1"]


def main():
    run_name_list = [
        # "random_chunked",
        # "location_chunked",
        "exact_match_chunked",
        # "w2v_chunked",
        # "coattention_chunked",
        "base_nli_chunked",
        "pep_chunked",
        "pep_nli_partial_chunked",
        "base_nli_partial_chunked"
        # "pep_word2vec_chunked"
    ]
    genre = "headlines"
    split = "train"
    show_scores(genre, run_name_list, split)


def main():
    run_name_list = [
        "base_w_context",
        "pep_w_context"
    ]

    genre = "images"
    split = "test"
    show_scores(genre, run_name_list, split)


def main_2():
    table = []
    is_first = True
    for split, genre in ists_enum_split_genre_combs():
        row = ["{}_{}".format(genre, split), ]
        head = ["dataset"]
        for nli_type in ["base", "pep", "em"]:
            for label_predictor_type in ["w_context", "wo_context"]:
                run_name = f"{nli_type}_{label_predictor_type}"
                head.append(run_name)
                save_path = get_ists_save_path(genre, split, run_name)
                if os.path.exists(save_path):
                    score = get_typed_f1(genre, split, run_name)
                    row.append(score)
                else:
                    row.append("-")
        if is_first:
            table.append(head)
            is_first = False
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main_2()
