import abc
import dataclasses
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



@dataclasses.dataclass
class TermPairPredictionConfig(HFModelConfigType):
    __metaclass__ = abc.ABCMeta
    max_seq_length = abc.abstractproperty()
    num_classes = abc.abstractproperty()
    model_type = abc.abstractproperty()
    n_mask_prepad = abc.abstractproperty()
    n_mask_postpad = abc.abstractproperty()


class ModelConfig64_1(HFModelConfigType):
    max_seq_length = 32
    num_classes = 1
    model_type = "bert-base-uncased"



class PredictionConfig32_1(TermPairPredictionConfig):
    max_seq_length = 32
    num_classes = 1
    model_type = "bert-base-uncased"
    n_mask_prepad = 4
    n_mask_postpad = 24


def get_term_pair_predictor_fixed_context(
        model_path,
        config: TermPairPredictionConfig,
) -> Callable[[List[Tuple[str, str]]], List[float]]:
    n_mask_prepad = config.n_mask_prepad
    n_mask_postpad = config.n_mask_postpad

    strategy = get_strategy()
    with strategy.scope():
        model = load_ts_concat_local_decision_model(config, model_path)
        pep = PEPLocalDecision(config, model_path=None, model=model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def score_term_pairs(term_pairs: List[Tuple[str, str]]) -> List[float]:
        payload = []
        info = []
        for q_term, d_term in term_pairs:
            q_tokens = tokenizer.tokenize(q_term)
            d_tokens = tokenizer.tokenize(d_term)
            q_tokens = ["[MASK]"] + q_tokens + ["[MASK]"]
            d_tokens = ["[MASK]"] * n_mask_prepad + d_tokens + ["[MASK]"] * n_mask_postpad

            info.append((q_term, d_term))
            payload.append((q_tokens, d_tokens))

        scores: List[float] = pep.score_fn(payload)
        return scores

    return score_term_pairs


def predict_with_fixed_context_model_and_save(
        config,
        model_path,
        log_path,
        candidate_itr: List[Tuple[str, str]],
        outer_batch_size,
        n_item=None
):
    predict_term_pairs = get_term_pair_predictor_fixed_context(model_path, config)
    predict_term_pairs_and_save(predict_term_pairs, candidate_itr, log_path, outer_batch_size, n_item)


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    candidate_pairs = read_term_pair_table(conf.table_path)
    num_items = len(candidate_pairs)
    model_path = conf.model_path
    log_path = conf.save_path
    config = PredictionConfig32_1()
    predict_with_fixed_context_model_and_save(
        config,
        model_path, log_path, candidate_pairs, 100, num_items)


if __name__ == "__main__":
    main()
