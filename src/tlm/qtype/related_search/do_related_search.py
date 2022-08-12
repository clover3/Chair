from typing import List, Tuple, Any

from alignment.data_structure import ContributionSummary
from alignment.data_structure.eval_data_structure import Alignment2D
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricWCIF
from tlm.qtype.partial_relevance.runner.related_prediction.run_search import get_one_hot_contribution


def related_search(
        p_list: List[RelatedEvalInstance],
        eval_policy: EvalMetricWCIF,
        preserve_idx) -> List[Alignment2D]:

    def get_predictions_for_case(a_p: Tuple[Alignment2D,
                                            RelatedEvalInstance
                                              ]):
        a, p = a_p
        return eval_policy.get_predictions_for_case(p, a, None)

    per_problem = []
    for p in p_list:
        dmc_list: List[int] = list(range(p.seg_instance.text2.get_seg_len()))
        dmc_paired_future = []
        for dmc in dmc_list:
            answer: Alignment2D = get_one_hot_contribution(p, preserve_idx, dmc)
            future = get_predictions_for_case((answer, p))
            dmc_paired_future.append((dmc, future))
        per_problem.append((p, dmc_paired_future))

    eval_policy.do_duty()

    def summarize(p: RelatedEvalInstance, dmc_paired: List[Tuple[Any, float]]) -> Alignment2D:
        score_d = dict(dmc_paired)
        scores = [score_d[idx] for idx in range(p.seg_instance.text2.get_seg_len())]
        cs = ContributionSummary.from_single_array(scores, preserve_idx, 2)
        return Alignment2D(p.problem_id, cs)

    output: List[Alignment2D] = []
    for p, dmc_paired_future in per_problem:
        dmc_paired = [(dmc, eval_policy.convert_future_to_score(future)) for dmc, future in dmc_paired_future]
        a: Alignment2D = summarize(p, dmc_paired)
        output.append(a)
    return output