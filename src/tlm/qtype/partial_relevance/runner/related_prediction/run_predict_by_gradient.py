from tlm.qtype.partial_relevance.attention_based.attention_mask_gradient import AttentionGradientScorer, get_attention_mask_predictor
from tlm.qtype.partial_relevance.runner.related_prediction.related_scoring_common import \
    load_problem_run_scoring_and_save


def get_attention_gradient_scorer() -> AttentionGradientScorer:
    max_seq_length = 512
    attention_mask_gradient = get_attention_mask_predictor()
    scorer = AttentionGradientScorer(attention_mask_gradient, max_seq_length)
    return scorer


def main():
    #   Inputs: (sentence pairs, possibly tokenized)
    #   Outputs: Score array. Let's go with json
    dataset_name = "dev"
    method = "gradient"
    scorer: AttentionGradientScorer = get_attention_gradient_scorer()
    load_problem_run_scoring_and_save(dataset_name, method, scorer)


if __name__ == "__main__":
    main()
