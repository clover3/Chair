from transformers import AutoTokenizer

from list_lib import apply_batch
from misc_lib import TimeEstimator, TimeEstimatorOpt
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.local_decision_helper import load_ts_concat_local_decision_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from typing import List, Iterable, Callable, Dict, Tuple, Set


def get_term_pair_predictor_fixed_context(
        model_path,
):
    strategy = get_strategy()
    with strategy.scope():
        model_config = ModelConfig256_1()
        model = load_ts_concat_local_decision_model(model_config, model_path)
        pep = PEPLocalDecision(model_config, model_path=None, model=model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def score_term_pairs(term_pairs: List[Tuple[str, str]]):
        payload = []
        info = []
        for q_term, d_term in term_pairs:
            q_tokens = tokenizer.tokenize(q_term)
            d_tokens = tokenizer.tokenize(d_term)
            q_tokens = ["[MASK]"] + q_tokens + ["[MASK]"]
            d_tokens = ["[MASK]"] * 4 + d_tokens + ["[MASK]"] * 24

            info.append((q_term, d_term))
            payload.append((q_tokens, d_tokens))

        scores: List[float] = pep.score_fn(payload)
        return scores

    return score_term_pairs


def get_term_pair_predictor_compress_mask(
        model_path,
):
    strategy = get_strategy()
    with strategy.scope():
        model_config = ModelConfig256_1()
        model = load_ts_concat_local_decision_model(model_config, model_path)
        pep = PEPLocalDecision(model_config, model_path=None, model=model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def score_term_pairs(term_pairs: List[Tuple[str, str]]):
        payload = []
        info = []
        for q_term, d_term in term_pairs:
            q_tokens = tokenizer.tokenize(q_term)
            d_tokens = tokenizer.tokenize(d_term)
            q_tokens = ["[MASK]"] + q_tokens + ["[MASK]"]
            d_tokens = ["[MASK]"] + d_tokens + ["[MASK]"]
            info.append((q_term, d_term))
            payload.append((q_tokens, d_tokens))

        scores: List[float] = pep.score_fn(payload)
        return scores

    return score_term_pairs


def predict_with_fixed_context_model_and_save(
        model_path,
        log_path,
        candidate_itr: List[Tuple[str, str]],
        outer_batch_size,
        n_item=None
):
    predict_term_pairs = get_term_pair_predictor_fixed_context(model_path)
    out_f = open(log_path, "w")

    if n_item is not None:
        n_batch = n_item // outer_batch_size
    else:
        n_batch = None

    ticker = TimeEstimatorOpt(n_batch)
    for batch in apply_batch(candidate_itr, outer_batch_size):
        scores = predict_term_pairs(batch)
        for (q_term, d_term), score in zip(batch, scores):
            out_f.write(f"{q_term}\t{d_term}\t{score}\n")
        ticker.tick()