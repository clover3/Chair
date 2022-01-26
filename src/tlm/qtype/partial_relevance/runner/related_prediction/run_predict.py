import sys
from typing import List

from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import AttentionMaskScorerIF
from tlm.qtype.partial_relevance.attention_based.attention_mask_gradient import AttentionGradientScorer, \
    get_attention_mask_gradient_predictor
from tlm.qtype.partial_relevance.attention_based.bert_mask_predictor import get_bert_mask_predictor
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import later_score_prob
from tlm.qtype.partial_relevance.attention_based.perturbation_scorer import PerturbationScorer
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.methdos.exact_match_scorer import ExactMatchScorer
from tlm.qtype.partial_relevance.methdos.random_score import RandomScorer
from tlm.qtype.partial_relevance.related_answer_data_path_helper import save_related_eval_answer
from tlm.qtype.partial_relevance.runner.related_prediction.related_scoring_common import run_scoring


def get_attn_mask_scorer() -> AttentionMaskScorerIF:
    # predictor = get_localhost_bert_mask_client()
    predictor = get_bert_mask_predictor()
    max_seq_length = 512
    scorer = PerturbationScorer(predictor, max_seq_length, later_score_prob)
    return scorer


def get_attention_gradient_scorer() -> AttentionGradientScorer:
    max_seq_length = 512
    attention_mask_gradient = get_attention_mask_gradient_predictor()
    scorer = AttentionGradientScorer(attention_mask_gradient, max_seq_length)
    return scorer


def get_method(method_name) -> AttentionMaskScorerIF:
    if method_name == "gradient":
        scorer: AttentionGradientScorer = get_attention_gradient_scorer()
    elif method_name == "attn_perturbation":
        scorer: AttentionMaskScorerIF = get_attn_mask_scorer()
    elif method_name == "random":
        scorer: AttentionMaskScorerIF = RandomScorer()
    elif method_name == "exact_match":
        scorer: AttentionMaskScorerIF = ExactMatchScorer()
    else:
        raise ValueError
    return scorer


def load_problem_run_scoring_and_save(dataset_name, method: str):
    print(f"load_problem_run_scoring_and_save(\"{dataset_name}\", \"{method}\")")
    scorer: AttentionMaskScorerIF = get_method(method)
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset_name)
    answers: List[RelatedEvalAnswer] = run_scoring(problems, scorer)
    save_related_eval_answer(answers, dataset_name, method)


def main():
    dataset_name = sys.argv[1]
    method = sys.argv[2]
    load_problem_run_scoring_and_save(dataset_name, method)


if __name__ == "__main__":
    main()
