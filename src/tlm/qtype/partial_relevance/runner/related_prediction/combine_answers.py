from tlm.qtype.partial_relevance.related_answer_data_path_helper import save_related_eval_answer, \
    load_related_eval_answer


def main():
    method = "gradient"
    all_answer = []
    for i in range(0, 10):
        dataset = "dev_word{}".format(i)
        answers = load_related_eval_answer(dataset, method)
        all_answer.extend(answers)

    save_related_eval_answer(all_answer, "dev_word", method)


if __name__ == "__main__":
    main()
