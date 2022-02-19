import sys
from typing import List

from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedBinaryAnswer, \
    PerProblemEvalResult
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_100_random_spans
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import TenStepRandomDropPolicy, \
    TenStepRandomReplacePolicy
from tlm.qtype.partial_relevance.eval_metric_binary.erasure_v2 import ErasureV2
from tlm.qtype.partial_relevance.eval_metric_binary.eval_conditional_v2_all import conditional_align_eval_replace, \
    conditional_align_eval_b
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v2 import ReplaceV2
from tlm.qtype.partial_relevance.eval_score_dp_helper import save_eval_result_b
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_binary_related_eval_answer


def hooking_log(items):
    n_item = len(items)
    time_estimate = 0.035 * n_item
    if time_estimate > 100:
        n_min = int(time_estimate / 60)
        print("hooking_log: {} items {} min".format(n_item, n_min))


def run_replace_v2(model_interface, problems, answers) -> List[PerProblemEvalResult]:
    client = get_mmd_cache_client(model_interface, hooking_log)
    forward_fn = client.predict
    replacee_span_list = get_100_random_spans()

    def get_word_pool(e) -> List[List[int]]:
        words = [span.ngram_ids for span in replacee_span_list]
        return words

    eval_policy = ReplaceV2(forward_fn, TenStepRandomReplacePolicy(), get_word_pool)
    return conditional_align_eval_replace(answers, problems, eval_policy)


def run_deletion_v2(model_interface, problems, answers) -> List[PerProblemEvalResult]:
    client = get_mmd_cache_client(model_interface, hooking_log)
    forward_fn = client.predict
    eval_policy = ErasureV2(forward_fn, TenStepRandomDropPolicy())
    return conditional_align_eval_b(answers, problems, eval_policy)


def run_align_eval_b(dataset, method, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset, method)
    print("{} problems {} answers".format(len(problems), len(answers)))
    if policy_name == "deletion_v2":
        scores: List[PerProblemEvalResult] = run_deletion_v2(model_interface, problems, answers)
    elif policy_name == "replace_v2":
        scores: List[PerProblemEvalResult] = run_replace_v2(model_interface, problems, answers)
    else:
        assert False
    run_name = "{}_{}_{}".format(dataset, method, policy_name)
    save_eval_result_b(scores, run_name)


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
