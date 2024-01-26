import sys
from typing import List, Callable, Tuple

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from trainer_v2.custom_loop.definitions import HFModelConfigType
from trainer_v2.per_project.transparency.misc_common import read_term_pair_table
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import predict_term_pairs_and_save
from trainer_v2.per_project.transparency.mmp.pep.local_decision_helper import load_ts_concat_local_decision_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy



class ModelConfig32_1(HFModelConfigType):
    max_seq_length = 32
    num_classes = 1
    model_type = "bert-base-uncased"


def get_term_pair_predictor_fixed_context(
        model_path,
) -> Callable[[List[Tuple[str, str]]], List[float]]:
    strategy = get_strategy()
    with strategy.scope():
        model_config = ModelConfig32_1()
        model = load_ts_concat_local_decision_model(model_config, model_path)
        pep = PEPLocalDecision(model_config, model_path=None, model=model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def score_term_pairs(term_pairs: List[Tuple[str, str]]) -> List[float]:
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


def predict_with_fixed_context_model_and_save(
        model_path,
        log_path,
        candidate_itr: List[Tuple[str, str]],
        outer_batch_size,
        n_item=None
):
    predict_term_pairs = get_term_pair_predictor_fixed_context(model_path)
    predict_term_pairs_and_save(predict_term_pairs, candidate_itr, log_path, outer_batch_size, n_item)


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    candidate_pairs = read_term_pair_table(conf.table_path)
    num_items = len(candidate_pairs)
    model_path = conf.model_path
    log_path = conf.save_path
    predict_with_fixed_context_model_and_save(
        model_path, log_path, candidate_pairs, 100, num_items)


if __name__ == "__main__":
    main()
