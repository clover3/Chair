from alignment.ists_eval.eval_helper import load_ists_predictions
from alignment.ists_eval.f1_calc import calc_f1
from dataset_specific.ists.parse import AlignmentPredictionList
from dataset_specific.ists.path_helper import load_ists_label


def main():
    gold: AlignmentPredictionList = load_ists_label("headlines", "train")
    run_name_list = ["random_one", "em", "word2vec", "coattn", "nlits", "nlits_mini50", "probe"]
    for run_name in run_name_list:
        mode = ""
        pred: AlignmentPredictionList = load_ists_predictions("headlines", "train", run_name)
        scores = calc_f1(gold, pred, mode)
        print("{}\t{}".format(run_name, scores['f1']))


if __name__ == "__main__":
    main()






