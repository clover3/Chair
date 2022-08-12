import xmlrpc.client
from typing import List

from alignment import MatrixScorerIF
from alignment.data_structure.eval_data_structure import Alignment2D
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from misc_lib import TEL


def run_scoring(problems: List[RelatedEvalInstance], scorer: MatrixScorerIF) -> List[Alignment2D]:
    answer_list: List[Alignment2D] = []
    for p in TEL(problems):
        try:
            c: ContributionSummary = scorer.eval_contribution(p.seg_instance)
            assert len(c.table[0]) == p.seg_instance.text2.get_seg_len()
        except xmlrpc.client.Fault:
            print(p.seg_instance.to_json())
            raise
        answer = Alignment2D(p.problem_id, c)
        answer_list.append(answer)
    return answer_list
