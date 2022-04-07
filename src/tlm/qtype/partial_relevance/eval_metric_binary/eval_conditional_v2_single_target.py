from typing import List, Tuple

from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from misc_lib import TEL
from contradiction.alignment.data_structure.eval_data_structure import RelatedBinaryAnswer, join_a_p
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricConditionalIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_partial_text_as_segment
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v2 import ReplaceV2SingleSeg


def conditional_align_eval_b(answer_list: List[RelatedBinaryAnswer],
                 problem_list: List[RelatedEvalInstance],
                 eval_policy: EvalMetricConditionalIF,
                 ) -> List[Tuple[str, float]]:
    a_p_list = join_a_p(answer_list, problem_list)

    conditions_future = [eval_policy.get_condition_pf(p, a) for a, p in a_p_list]
    eval_policy.do_duty()
    applicable_list: List[bool] = list(map(eval_policy.convert_condition_pf, conditions_future))
    problem_ids: List[str] = [p.problem_id for a, p in a_p_list]

    future_predictions_list = []
    applicable_problem_ids = []
    for idx in range(len(a_p_list)):
        if applicable_list[idx]:
            a, p = a_p_list[idx]
            future = eval_policy.get_test_pf(p, a)
            future_predictions_list.append(future)
            applicable_problem_ids.append(problem_ids[idx])

    eval_policy.do_duty()
    eval_score_list: List[float] = list(map(eval_policy.convert_test_pf, future_predictions_list))
    problem_id_to_score = dict(zip(applicable_problem_ids, eval_score_list))

    def get_score(problem_id: str):
        try:
            return problem_id_to_score[problem_id]
        except KeyError:
            return None

    scores_list_w_none = list(map(get_score, problem_ids))
    return list(zip(problem_ids, scores_list_w_none))


def conditional_align_eval_replace(answer_list: List[RelatedBinaryAnswer],
                                   problem_list: List[RelatedEvalInstance],
                                   eval_policy: ReplaceV2SingleSeg,
                                   ) -> List[Tuple[str, float]]:
    a_p_list = join_a_p(answer_list, problem_list)
    tokenizer = get_tokenizer()
    output = []
    for a, p in TEL(a_p_list):
        # print(rei_to_text(tokenizer, p))
        conditions_future = eval_policy.get_condition_pf(p, a)
        eval_policy.do_duty()
        found, pw_found = eval_policy.convert_condition_pf(conditions_future)
        qt: SegmentedText = get_partial_text_as_segment(p.seg_instance.text1, 1)
        qt_s = pretty_tokens(tokenizer.convert_ids_to_tokens(qt.tokens_ids))
        # print("qt: ", qt_s)
        if found:
            test_pf = eval_policy.get_test_pf(p, a, pw_found)
            eval_policy.do_duty()
            score: float = eval_policy.convert_test_pf(test_pf)
            # print("found")
        else:
            # print("Not found")
            score = None
        output.append((p.problem_id, score))
    return output

