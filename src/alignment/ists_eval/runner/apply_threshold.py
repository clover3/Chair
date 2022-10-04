from alignment.ists_eval.threshold_util import apply_threshold_and_save


def main():
    run_name = "coattn"
    genre = "headlines"
    split = "test"
    apply_threshold_and_save(genre, run_name, "train")
    apply_threshold_and_save(genre, run_name, "test")


if __name__ == "__main__":
    main()
