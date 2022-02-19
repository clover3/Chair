import xmlrpc.client
from typing import List

from misc_lib import TEL
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, ContributionSummary, \
    MatrixScorerIF


def run_scoring(problems: List[RelatedEvalInstance], scorer: MatrixScorerIF) -> List[RelatedEvalAnswer]:
    answer_list: List[RelatedEvalAnswer] = []
    for p in TEL(problems):
        try:
            c: ContributionSummary = scorer.eval_contribution(p.seg_instance)
            assert len(c.table[0]) == p.seg_instance.text2.get_seg_len()
        except xmlrpc.client.Fault:
            print(p.seg_instance.to_json())
            raise
        answer = RelatedEvalAnswer(p.problem_id, c)
        answer_list.append(answer)
    return answer_list
