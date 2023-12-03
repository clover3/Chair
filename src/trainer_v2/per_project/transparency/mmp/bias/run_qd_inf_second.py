import sys

from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list, \
    load_second_column_from_bias_dir
from typing import List
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.bias.inference_w_keyword_swap import run_inference_inner
from cpath import output_path
from misc_lib import path_join, TimeEstimator, two_digit_float
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bias.common import find_indices
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def load_car_bias_exp_resource_dev():
    term_list = load_car_maker_list()
    term_list_set = set(term_list)

    query_list: List[str] = load_second_column_from_bias_dir("dev_matched_car_queries_no_maker.tsv")
    passages = load_second_column_from_bias_dir("matched_doc_dev_all.tsv")
    return passages, query_list, term_list_set, term_list


def run_inference_inner(doc_iter, model_path, query_list, term_list_set, term_list, score_log_f):
    batch_size = 256

    def car_maker_replace(text):
        tokens = text.split()
        matching_indices = find_indices(tokens, term_list_set)
        if not matching_indices:
            print("WARNING: {} does not have matching tokens".format(text))

        for term in term_list:
            tokens_new = list(tokens)
            for i in matching_indices:
                tokens_new[i] = term
            yield " ".join(tokens_new)

    strategy = get_strategy()
    with strategy.scope():
        c_log.info("Building scorer")
        score_fn = get_scorer(model_path, batch_size)
        candidates = []
        for doc_idx, doc_text in doc_iter():
            payload = [(query, doc_text) for query in query_list]
            scores = score_fn(payload)
            for q_idx, s in enumerate(scores):
                if s > 0:
                    candidates.append((doc_idx, q_idx))
                    row = [str(doc_idx), str(q_idx)]
                    score_log_f.write("\t".join(row) + "\n")
                    score_log_f.flush()


def main():
    model_path = sys.argv[1]
    job_no = int(sys.argv[2])

    passages, query_list, term_list_set, term_list = load_car_bias_exp_resource_dev()

    n_per_job = 10
    score_log_f = open(path_join(output_path, "mmp", "bias", "score_log_dev_pre", f"{job_no}.tsv"), "w")
    st = n_per_job * job_no
    ed = st + n_per_job

    def doc_iter():
        for doc_idx in range(st, ed):
            doc_text = passages[doc_idx]
            yield doc_idx, doc_text

    run_inference_inner(doc_iter, model_path, query_list, term_list_set, term_list, score_log_f)


if __name__ == "__main__":
    main()
