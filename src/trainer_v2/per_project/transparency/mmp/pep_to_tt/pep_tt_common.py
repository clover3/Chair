import os
from collections import defaultdict
from typing import Dict, List, Iterable

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from list_lib import left, apply_batch
from misc_lib import path_join, get_second, TimeEstimatorOpt
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import HFModelConfigType, ModelConfig512_1
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel
from trainer_v2.per_project.transparency.misc_common import read_lines, save_tsv
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def load_local_decision_only_model(
        new_model_config: HFModelConfigType, model_save_path):
    task_model = TwoSegConcatLogitCombineTwoModel(
        new_model_config, CombineByScoreAdd)
    task_model.build_model(None)
    model = task_model.define_local_only_model()
    if model_save_path == "skip":
        c_log.info("Skip model loading. ")
    else:
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.restore(model_save_path).expect_partial()

    return model


def build_term_pair_scorer(model_path):
    strategy = get_strategy()
    with strategy.scope():
        src_model_config = ModelConfig512_1()
        model = load_local_decision_only_model(src_model_config, model_path)
        max_seq_length = src_model_config.max_seq_length // 2

    tokenizer = get_tokenizer()
    def predict_one(term1, term2):
        tokens1 = tokenizer.tokenize(term1)
        tokens2 = tokenizer.tokenize(term2)
        input_ids, segment_ids = combine_with_sep_cls_and_pad(
            tokenizer, tokens1, tokens2, max_seq_length)

        def reform(ids):
            t = tf.constant(ids)
            return tf.expand_dims(t, axis=0)

        x = reform(input_ids), reform(segment_ids)
        y = model(x)
        return y[0]

    return model, predict_one


def load_term_pair_table(conf) -> Dict[str, Dict[str, float]]:
    save_dir = conf.score_save_dir
    q_terms = read_lines(conf.q_term_path)
    n_pair = 0
    output_d: Dict[str, Dict[str, float]] = defaultdict(dict)
    for q_term_i in range(len(q_terms)):
        log_path = path_join(save_dir, f"{q_term_i}.txt")
        if not os.path.exists(log_path):
            continue
        q_term = q_terms[q_term_i]
        d_term_scores = [(row[0], float(row[1])) for row in tsv_iter(log_path)]
        d_term_scores_d = dict(d_term_scores)
        output_d[q_term] = d_term_scores_d
        n_pair += len(d_term_scores_d)

    c_log.info("Loaded %d pairs over %d terms", n_pair, len(q_terms))
    return output_d


def get_pep_predictor_dummy(conf):
    def get_pep_top_k(q_term, d_term_iter) -> List[str]:
        return ["is", "the"]
    return get_pep_top_k


def get_pep_predictor(conf):
    score_d: Dict[str, Dict[str, float]] = load_term_pair_table(conf)
    def get_pep_top_k(q_term, d_term_iter) -> List[str]:
        score_d_per_q_term = score_d[q_term]
        # if len(score_d_per_q_term) == 0:
        #     c_log.info(f"Entries for query term %s not found", q_term)
        d_term_list = list(d_term_iter)
        scores = []
        for d_term in d_term_list:
            try:
                scores.append((d_term, score_d_per_q_term[d_term]))
            except KeyError:
                pass

        scores.sort(key=get_second, reverse=True)
        return left(scores)
    return get_pep_top_k


def predict_pairs_save(
        predict_term_pairs_fn,
        term_pair_iter: Iterable[tuple[str, str]],
        log_path,
        outer_batch_size):

    if os.path.exists(log_path):
        print(f"{log_path} exists ")
        return

    if isinstance(term_pair_iter, list):

        n_item = len(term_pair_iter)
        n_batch = n_item // outer_batch_size
    else:
        n_batch = None

    ticker = TimeEstimatorOpt(n_batch)
    save_items = []
    for batch_pairs in apply_batch(term_pair_iter, outer_batch_size):
        scores = predict_term_pairs_fn(batch_pairs)

        for (q_term, d_term), score in zip(batch_pairs, scores):
            save_items.append((q_term, d_term, score))

        ticker.tick()
    save_tsv(save_items, log_path)
