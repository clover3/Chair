from list_lib import index_by_fn
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricIF
from trainer.promise import MyPromise, PromiseKeeper
from typing import List, Callable, Dict, Tuple


#  candidate metric
#   1. Area Under Reserve/Erasure Curve
#   2. Top-k drop flip.


def related_eval(answer_list: List[RelatedEvalAnswer],
                 problem_list: List[RelatedEvalInstance],
                 forward_fn: Callable[[List[SegmentedInstance]], List[float]],
                 drop_rate
                 ) -> List[float]:
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problem_list)
    pk = PromiseKeeper(forward_fn)

    def get_predictions_for_problem(answer, problem):
        future_prediction_list = []
        for seg_idx in range(problem.seg_instance.text1.get_seg_len()):
            new_text1 = get_top_k_drop_pair(answer, problem, drop_rate, seg_idx)
            new_seg = SegmentedInstance(new_text1, problem.seg_instance.text2)
            future_prediction = MyPromise(new_seg, pk).future()
            future_prediction_list.append(future_prediction)
        return future_prediction_list

    payloads = []
    for answer in answer_list:
        problem: RelatedEvalInstance = pid_to_p[answer.problem_id]
        future_prediction_list = get_predictions_for_problem(answer, problem)
        payloads.append((problem, future_prediction_list))

    pk.do_duty(True)

    eval_score_list = []
    for problem, future_prediction_list in payloads:
        for idx, future_prediction in enumerate(future_prediction_list):
            new_prediction = future_prediction.get()
            original_prediction = problem.score
            eval_score = original_prediction - new_prediction  # Higher the better
            print("{0:.2f} -> {1:.2f}".format(original_prediction, new_prediction))
            eval_score_list.append(eval_score)
    return eval_score_list


def partial_related_eval(answer_list: List[RelatedEvalAnswer],
                         problem_list: List[RelatedEvalInstance],
                         complement_list: List[ComplementSearchOutput],
                         eval_policy: EvalMetricIF,
                         ) -> List[Tuple[str, float]]:
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problem_list)
    pid_to_c: Dict[str, ComplementSearchOutput] = index_by_fn(lambda e: e.problem_id, complement_list)
    a_p_c_list: List[Tuple[RelatedEvalAnswer, RelatedEvalInstance, ComplementSearchOutput]] = []
    for a in answer_list:
        p: RelatedEvalInstance = pid_to_p[a.problem_id]
        c: ComplementSearchOutput = pid_to_c[a.problem_id]
        a_p_c_list.append((a, p, c))

    return partial_related_eval_inner(a_p_c_list, eval_policy)


def partial_related_eval_inner(
        a_p_c_list: List[Tuple[RelatedEvalAnswer,
                               RelatedEvalInstance,
                               ComplementSearchOutput]],
        eval_policy: EvalMetricIF,
        ) -> List[Tuple[str, float]]:

    def get_predictions_for_case(a_p_c: Tuple[RelatedEvalAnswer,
                                              RelatedEvalInstance,
                                              ComplementSearchOutput]):
        answer, problem, complement = a_p_c
        return eval_policy.get_predictions_for_case(problem, answer, complement)

    problem_ids: List[str] = [p.problem_id for a, p, c in a_p_c_list]
    future_predictions_list = list(map(get_predictions_for_case, a_p_c_list))
    eval_policy.do_duty()
    eval_score_list: List[float] = list(map(eval_policy.convert_future_to_score, future_predictions_list))
    return list(zip(problem_ids, eval_score_list))

