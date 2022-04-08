import sys
from typing import List
from typing import Tuple

from contradiction.alignment.data_structure.eval_data_structure import RelatedBinaryAnswer
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_100_random_spans
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import TenStepRandomDropPolicy, \
    TenStepRandomReplacePolicy
from tlm.qtype.partial_relevance.eval_metric_binary.erasure_v2 import ErasureV2_single_seg
from tlm.qtype.partial_relevance.eval_metric_binary.eval_conditional_v2_single_target import conditional_align_eval_b, \
    conditional_align_eval_replace
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v2 import ReplaceV2SingleSeg
from tlm.qtype.partial_relevance.eval_score_dp_helper import save_eval_result_b_single
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_binary_related_eval_answer


def run_replace_v2(target_seg_idx, model_interface, problems, answers) -> List[Tuple[str, float]]:
    client = get_mmd_cache_client(model_interface)
    forward_fn = client.predict
    replacee_span_list = get_100_random_spans()

    def get_word_pool(e) -> List[List[int]]:
        words = [span.ngram_ids for span in replacee_span_list]
        return words

    eval_policy = ReplaceV2SingleSeg(forward_fn, TenStepRandomReplacePolicy(), target_seg_idx, get_word_pool)
    return conditional_align_eval_replace(answers, problems, eval_policy)


def run_deletion_v2(target_seg_idx, model_interface, problems, answers) -> List[Tuple[str, float]]:
    client = get_mmd_cache_client(model_interface)
    forward_fn = client.predict
    eval_policy = ErasureV2_single_seg(forward_fn,
                                       TenStepRandomDropPolicy(),
                                       target_seg_idx=target_seg_idx)
    return conditional_align_eval_b(answers, problems, eval_policy)


def run_align_eval_b(dataset, method, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset, method)
    print("{} problems {} answers".format(len(problems), len(answers)))
    target_seg_idx = 1
    if policy_name == "deletion_v2":
        scores: List[Tuple[str, float]] = run_deletion_v2(target_seg_idx, model_interface, problems, answers)
    elif policy_name == "replace_v2":
        scores: List[Tuple[str, float]] = run_replace_v2(target_seg_idx, model_interface, problems, answers)
    else:
        assert False
    run_name = "{}_{}_{}".format(dataset, method, policy_name)
    save_eval_result_b_single(scores, run_name)


def main():
    dataset = sys.argv[1]
    method = sys.argv[2]
    policy_name = sys.argv[3]
    if len(sys.argv) > 4:
        model_interface = sys.argv[4]
    else:
        model_interface = "localhost"

    run_align_eval_b(dataset, method, policy_name, model_interface)


if __name__ == "__main__":
    main()
