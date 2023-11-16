from transformers import AutoTokenizer

from list_lib import apply_batch
from misc_lib import TimeEstimator
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.local_decision_helper import load_ts_concat_local_decision_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def predict_with_fixed_context_model(
        model_path,
        log_path,
        candidate_itr,
        outer_batch_size,
        n_item=None
):
    strategy = get_strategy()
    with strategy.scope():
        model_config = ModelConfig256_1()
        model = load_ts_concat_local_decision_model(model_config, model_path)
        pep = PEPLocalDecision(model_config, model_path=None, model=model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    out_f = open(log_path, "w")

    if n_item is not None:
        n_batch = n_item // outer_batch_size
        ticker = TimeEstimator(n_batch)
    for batch in apply_batch(candidate_itr, outer_batch_size):
        payload = []
        info = []
        for q_term, d_term in batch:
            q_tokens = tokenizer.tokenize(q_term)
            d_tokens = tokenizer.tokenize(d_term)
            q_tokens = ["[MASK]"] + q_tokens + ["[MASK]"]
            d_tokens = ["[MASK]"] * 4 + d_tokens + ["[MASK]"] * 24

            info.append((q_term, d_term))
            payload.append((q_tokens, d_tokens))

        scores = pep.score_fn(payload)

        for (q_term, d_term), score in zip(info, scores):
            out_f.write(f"{q_term}\t{d_term}\t{score}\n")
        if n_item is not None:
            ticker.tick()