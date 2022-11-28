from alignment.ists_eval.f1_calc import calc_f1
from alignment.ists_eval.matrix_eval_helper import load_ists_predictions
from dataset_specific.ists.parse import AlignmentPredictionList
from dataset_specific.ists.path_helper import load_ists_label
from tab_print import print_table


def show_scores(genre, run_name_list, split):
    gold: AlignmentPredictionList = load_ists_label(genre, split)
    mode_list = ["", "type", "score"]
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


def main():
    run_name_list = [
        # "random_chunked",
        # "location_chunked",
        # "exact_match_chunked",
        # "w2v_chunked",
        # "coattention_chunked",
        # "base_nli_chunked",
        # "pep_chunked",
        "pep_nli_partial_chunked",
        "base_nli_partial_chunked"
        # "pep_word2vec_chunked"
    ]
    genre = "headlines"
    split = "train"
    show_scores(genre, run_name_list, split)


if __name__ == "__main__":
    main()
