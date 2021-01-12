from arg.perspectives.w_new_candidates.common import qck_gen_w_ranked_list


def main():
    split = "train"
    job_name = "qck5"
    qk_candidate_name = "pc_qk2_train_filtered"
    ranked_list_path = "train"
    # Selected from doc_scorer_summarizer.py
    qck_gen_w_ranked_list(job_name, qk_candidate_name, ranked_list_path, split)


if __name__ == "__main__":
    main()
