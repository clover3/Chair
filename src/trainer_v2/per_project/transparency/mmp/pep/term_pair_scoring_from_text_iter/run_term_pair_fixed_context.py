

import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from iter_util import load_jsonl
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.local_decision_helper import load_ts_concat_local_decision_model
from trainer_v2.per_project.transparency.mmp.pep.term_pair_scoring_from_text_iter.read_segment_log import load_segment_log
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def predict_with_fixed_context_model(
        model_path,
        log_path,
        score_d,
        n=2000):
    strategy = get_strategy()
    with strategy.scope():
        model_config = ModelConfig256_1()
        model = load_ts_concat_local_decision_model(model_config, model_path)
        pep = PEPLocalDecision(model_config, model_path=None, model=model)

    tokenizer = get_tokenizer()
    payload = []
    info = []
    for q_term, d_term in score_d:
        q_tokens = tokenizer.tokenize(q_term)
        d_tokens = tokenizer.tokenize(d_term)

        q_tokens = ["[MASK]"] + q_tokens + ["[MASK]"]
        d_tokens = ["[MASK]"] * 4 + d_tokens + ["[MASK]"] * 24

        info.append((q_term, d_term))
        payload.append((q_tokens, d_tokens))
        if len(payload) >= n:
            break

    scores = pep.score_fn(payload)

    out_f = open(log_path, "w")
    for (q_term, d_term), score in zip(info, scores):
        out_f.write(f"{q_term}\t{d_term}\t{score}\n")


def main():
    jsonl = load_jsonl(sys.argv[1])
    score_d = load_segment_log(jsonl)

    model_path = sys.argv[2]
    log_path = sys.argv[3]
    predict_with_fixed_context_model(model_path, log_path, score_d, 200)


if __name__ == "__main__":
    main()
