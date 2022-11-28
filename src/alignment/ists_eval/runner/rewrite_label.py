from alignment.ists_eval.eval_utils import save_ists_predictions
from alignment.ists_eval.path_helper import get_ists_save_path
from dataset_specific.ists.parse import AlignmentPredictionList, AlignmentLabelUnit, parse_label_file
from dataset_specific.ists.path_helper import load_ists_label


def sort_key(a: AlignmentLabelUnit):
    v = a.chunk_token_id1[0]
    if v == 0:
        return 99999
    else:
        return v


def main():
    genre = "headlines"
    split = "train"
    gold: AlignmentPredictionList = load_ists_label(genre, split)

    do_sort(gold)

    run_name = "gold"
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(gold, save_path)


def do_sort(gold):
    for name, items in gold:
        items.sort(key=sort_key)


def main():
    genre = "headlines"
    split = "train"
    run_name = "pep_word2vec_chunked"
    save_path = get_ists_save_path(genre, split, run_name)
    preds = parse_label_file(save_path)
    do_sort(preds)

    save_run_name = run_name + "_sorted"
    save_path = get_ists_save_path(genre, split, save_run_name)
    save_ists_predictions(preds, save_path)


if __name__ == "__main__":
    main()