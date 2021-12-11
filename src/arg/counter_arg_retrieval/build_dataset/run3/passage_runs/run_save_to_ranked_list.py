from arg.counter_arg_retrieval.build_dataset.run3.swtt.save_trec_style import read_pickled_predictions_and_save


def save_to_ranked_list():
    run_name = "PQ_3"
    read_pickled_predictions_and_save(run_name)


def main():
    # read_pickled_predictions_and_save("PQ_5")
    read_pickled_predictions_and_save("PQ_2")
    # read_pickled_predictions_and_save("PQ_4")


if __name__ == "__main__":
    main()