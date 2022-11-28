from alignment.ists_eval.eval_utils import save_ists_predictions
from alignment.ists_eval.path_helper import get_ists_save_path
from dataset_specific.ists.parse import AlignmentPredictionList, AlignmentLabelUnit
from dataset_specific.ists.path_helper import load_ists_label


def main():
    genre = "headlines"
    split = "train"
    run_name = "gold"
    gold: AlignmentPredictionList = load_ists_label(genre, split)

    for _, aligns in gold:
        aligns.sort(key=AlignmentLabelUnit.sort_key)
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(gold, save_path)


if __name__ == "__main__":
    main()