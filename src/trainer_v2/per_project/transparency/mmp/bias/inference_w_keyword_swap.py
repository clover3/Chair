from misc_lib import path_join, TimeEstimator, two_digit_float
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bias.common import find_indices
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer
from trainer_v2.train_util.get_tpu_strategy import get_strategy


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
        for doc_idx, doc_text in doc_iter():
            text_list = list(car_maker_replace(doc_text))
            for q_idx, query in enumerate(query_list):
                tuple_itr = [(query, t) for t in text_list]
                scores = score_fn(tuple_itr)
                row = [str(doc_idx), str(q_idx)] + list(map(two_digit_float, scores))
                score_log_f.write("\t".join(row) + "\n")
                score_log_f.flush()


def run_inference_inner2(qd_iter, model_path, term_list_set, term_list, score_log_f):
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
        for entry in qd_iter:
            doc_text = entry["doc_text"]
            query_text = entry["query"]
            text_list = list(car_maker_replace(doc_text))
            tuple_itr = [(query_text, t) for t in text_list]
            scores = score_fn(tuple_itr)
            row = [str(entry["qid"]), str(entry["doc_id"])] + list(map(two_digit_float, scores))
            score_log_f.write("\t".join(row) + "\n")
            score_log_f.flush()


