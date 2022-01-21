import xmlrpc.client
from typing import List

from misc_lib import TEL
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import AttentionMaskScorerIF
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, ContributionSummary
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import save_related_eval_answer


def run_scoring(problems: List[RelatedEvalInstance], scorer: AttentionMaskScorerIF) -> List[RelatedEvalAnswer]:
    answer_list: List[RelatedEvalAnswer] = []
    for p in TEL(problems):
        try:
            c: ContributionSummary = scorer.eval_contribution(p.seg_instance)
        except xmlrpc.client.Fault:
            print(p.seg_instance.to_json())
            raise
        answer = RelatedEvalAnswer(p.problem_id, c)
        answer_list.append(answer)
    return answer_list


def load_problem_run_scoring_and_save(dataset_name, method, scorer):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset_name)
    answers: List[RelatedEvalAnswer] = run_scoring(problems, scorer)
    save_related_eval_answer(answers, dataset_name, method)