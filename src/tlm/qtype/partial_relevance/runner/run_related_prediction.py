import json
import os
from typing import List

from cpath import output_path
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import AttentionMaskScorerIF
from tlm.qtype.partial_relevance.attention_based.attention_mask_gradient import PredictorAttentionMaskGradient, \
    AttentionGradientScorer
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, ContributionSummary, RelatedEvalAnswer
from tlm.qtype.partial_relevance.loader import load_dev_small_problems


def run_scoring(problems: List[RelatedEvalInstance], scorer: AttentionMaskScorerIF) -> List[RelatedEvalAnswer]:
    answer_list: List[RelatedEvalAnswer] = []
    for p in problems:
        c: ContributionSummary = scorer.eval_contribution(p.seg_instance)
        answer = RelatedEvalAnswer(p.problem_id, c)
        answer_list.append(answer)
    return answer_list


def get_attention_gradient_scorer() -> AttentionGradientScorer:
    max_seq_length = 512
    attention_mask_gradient = PredictorAttentionMaskGradient(2, max_seq_length)
    scorer = AttentionGradientScorer(attention_mask_gradient, max_seq_length)
    return scorer


def main():
    #   Inputs: (sentence pairs, possibly tokenized)
    #   Outputs: Score array. Let's go with json
    problems: List[RelatedEvalInstance] = load_dev_small_problems()
    scorer: AttentionGradientScorer = get_attention_gradient_scorer()
    answers = run_scoring(problems, scorer)
    save_path = os.path.join(output_path, "qtype", "related_scores", "MMDE_dev_mmd_Z.score")
    json.dump(answers, open(save_path, "w"), indent=True)


if __name__ == "__main__":
    main()
