from alignment.ists_eval.matrix_eval_helper import load_ists_predictions
from alignment.ists_eval.f1_calc import calc_f1
from dataset_specific.ists.parse import AlignmentPredictionList
from dataset_specific.ists.path_helper import load_ists_label
from tab_print import print_table


def main():
    gold: AlignmentPredictionList = load_ists_label("headlines", "train")
    run_name_list = ["random_one", "em", "word2vec", "coattn", "nlits", "nlits_mini50", "probe"]
    head = ["method", "f1", "precision", "recall"]
    table = [head]
    for run_name in run_name_list:
        mode = ""
        pred: AlignmentPredictionList = load_ists_predictions("headlines", "train", run_name)
        scores = calc_f1(gold, pred, mode)
        row = [run_name, scores['f1'], scores['precision'], scores['recall']]
        table.append(row)

    print_table(table)


def main2():
    split_list = ["train", "test"]
    run_name = "coattn"
    for split in split_list:
        gold: AlignmentPredictionList = load_ists_label("headlines", split)
        head = ["method", "f1", "precision", "recall"]
        table = [head]
        mode = ""
        pred: AlignmentPredictionList = load_ists_predictions("headlines", split, run_name)
        scores = calc_f1(gold, pred, mode)
        row = [run_name, scores['f1'], scores['precision'], scores['recall']]
        table.append(row)
        print_table(table)


if __name__ == "__main__":
    main2()






