from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import AttentionMaskScorerIF
from tlm.qtype.partial_relevance.attention_based.bert_masking_client import get_localhost_bert_mask_client
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import later_score_prob
from tlm.qtype.partial_relevance.attention_based.perturbation_scorer import PerturbationScorer
from tlm.qtype.partial_relevance.runner.related_prediction.related_scoring_common import load_problem_run_scoring_and_save


def get_scorer() -> AttentionMaskScorerIF:
    predictor = get_localhost_bert_mask_client()
    max_seq_length = 512
    scorer = PerturbationScorer(predictor, max_seq_length, later_score_prob)
    return scorer


def main():
    dataset_name = "dev"
    method = "attn_perturbation"
    scorer: AttentionMaskScorerIF = get_scorer()
    load_problem_run_scoring_and_save(dataset_name, method, scorer)


if __name__ == "__main__":
    main()
