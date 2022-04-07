from typing import List, Tuple, Iterable, Optional

from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from list_lib import left, right
from misc_lib import TEL
from contradiction.alignment.data_structure.eval_data_structure import RelatedBinaryAnswer, join_a_p, \
    RelatedEvalInstanceEx, PerProblemEvalResult
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricConditionalPerTargetIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_partial_text_as_segment
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v2 import ReplaceV2
from trainer.promise import parent_child_pattern, MyFuture


def unpack_problem(p: RelatedEvalInstance) -> List[RelatedEvalInstanceEx]:
    output: List[RelatedEvalInstanceEx] = []
    for i in range(p.seg_instance.text1.get_seg_len()):
        reix = RelatedEvalInstanceEx(p.problem_id, i, p.seg_instance, p.score)
        output.append(reix)
    return output


def conditional_align_eval_b(answer_list: List[RelatedBinaryAnswer],
                             problem_list: List[RelatedEvalInstance],
                             eval_policy: EvalMetricConditionalPerTargetIF,
                             ) -> List[PerProblemEvalResult]:
    a_p_list = join_a_p(answer_list, problem_list)
    TestOutputType = float
    Parent = Tuple[RelatedBinaryAnswer, RelatedEvalInstance]
    Child = Tuple[RelatedBinaryAnswer, RelatedEvalInstanceEx]

    def unpack_problem_ex(a_p: Tuple[RelatedBinaryAnswer, RelatedEvalInstance])\
            -> List[Tuple[RelatedBinaryAnswer, RelatedEvalInstanceEx]]:
        a, p = a_p
        return [(a, c) for c in unpack_problem(p)]

    def work_for_children(children: List[Child]) -> List[Optional[TestOutputType]]:
        future_list: List[MyFuture] = [eval_policy.get_condition_pf(p, a) for a, p in children]
        eval_policy.do_duty()
        applicable_list: List[bool] = [eval_policy.convert_condition_pf(pf) for pf in future_list]
        applicable_indices: List[int] = [i for i in range(len(children)) if applicable_list[i]]
        itr_applicable_children: Iterable[Child] = [children[i] for i in applicable_indices]
        test_pf_list: List[MyFuture[TestOutputType]] = [eval_policy.get_test_pf(p, a) for a, p in itr_applicable_children]
        eval_policy.do_duty()
        test_output_list: List[TestOutputType] = [eval_policy.convert_test_pf(pf) for pf in test_pf_list]
        output: List[Optional[TestOutputType]] = [None for _ in range(len(children))]
        for v, idx in zip(test_output_list, applicable_indices):
            output[idx] = v
        return output

    ltl: List[Tuple[Parent, List[Optional[TestOutputType]]]] = parent_child_pattern(a_p_list,
                                                                                    unpack_problem_ex,
                                                                                    work_for_children)
    problem_ids: List[str] = [p.problem_id for a, p in left(ltl)]
    return [PerProblemEvalResult(pid, res) for pid, res in zip(problem_ids, right(ltl))]


def conditional_align_eval_replace(answer_list: List[RelatedBinaryAnswer],
                                   problem_list: List[RelatedEvalInstance],
                                   eval_policy: ReplaceV2,
                                   ) -> List[PerProblemEvalResult]:
    a_p_list: List[Tuple[RelatedBinaryAnswer, RelatedEvalInstance]] = join_a_p(answer_list, problem_list)
    tokenizer = get_tokenizer()
    output: List[PerProblemEvalResult] = []
    for a, p in TEL(a_p_list):
        # print(rei_to_text(tokenizer, p))
        children = unpack_problem(p)
        scores: List[Optional[float]] = []
        for p in children:
            conditions_future = eval_policy.get_condition_pf(p, a)
            eval_policy.do_duty()
            found, pw_found = eval_policy.convert_condition_pf(conditions_future)
            qt: SegmentedText = get_partial_text_as_segment(p.seg_instance.text1, 1)
            qt_s = pretty_tokens(tokenizer.convert_ids_to_tokens(qt.tokens_ids))
            # print("qt: ", qt_s)
            score: Optional[float] = None
            if found:
                test_pf = eval_policy.get_test_pf(p, a, pw_found)
                eval_policy.do_duty()
                score = eval_policy.convert_test_pf(test_pf)
                # print("found")
            else:
                # print("Not found")
                score = None
            scores.append(score)
        output.append(PerProblemEvalResult(p.problem_id, scores))
    return output

