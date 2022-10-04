from alignment.ists_eval.threshold_util import apply_threshold_and_save, threshold_tuning


def main():
    run_name = "nlits"
    genre = "headlines"
    split = "train"
    threshold_tuning(genre, run_name, split)


if __name__ == "__main__":
    main()
