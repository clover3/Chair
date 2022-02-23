import sys
from typing import List

from tlm.qtype.partial_relevance.bert_mask_interface.mmd_z_mask_cacche import get_attn_mask_forward_fn, AttnMaskForward
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedBinaryAnswer, \
    PerProblemEvalResult, UnexpectedPolicyException
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_100_random_spans
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import TenStepRandomReplacePolicy, \
    TenStepRandomDropPolicy
from tlm.qtype.partial_relevance.eval_metric_binary.attn_v3 import AttnEvalMetric
from tlm.qtype.partial_relevance.eval_metric_binary.erasure_v3 import ErasureV3, ErasureSufficiency
from tlm.qtype.partial_relevance.eval_metric_binary.eval_v3 import eval_v3
from tlm.qtype.partial_relevance.eval_metric_binary.eval_v3_common import EvalMetricV3IF, MetricV3
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v3 import ReplaceV3, ReplaceV3S
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v31 import ReplaceV31
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v32 import ReplaceV32
from tlm.qtype.partial_relevance.eval_score_dp_helper import save_eval_result_b, get_run_name
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_binary_related_eval_answer


def hooking_log(items):
    n_item = len(items)
    time_estimate = 0.035 * n_item
    if time_estimate > 100:
        n_min = int(time_estimate / 60)
        print("hooking_log: requesting {} items {} min expected".format(n_item, n_min))


def get_replace_v3(model_interface, policy_name) -> MetricV3:
    client = get_mmd_cache_client(model_interface, hooking_log)
    forward_fn = client.predict
    replacee_span_list = get_100_random_spans()

    def get_word_pool(e) -> List[List[int]]:
        words = [span.ngram_ids for span in replacee_span_list]
        return words
    if policy_name == 'replace_v3':
        return ReplaceV3(forward_fn, TenStepRandomReplacePolicy(), get_word_pool)
    elif policy_name == 'replace_v3d':
        return ReplaceV3(forward_fn, TenStepRandomReplacePolicy(True), get_word_pool)
    elif policy_name == "replace_v31":
        return ReplaceV31(forward_fn, TenStepRandomReplacePolicy(), get_word_pool)
    elif policy_name == "replace_v32":
        return ReplaceV32(forward_fn, TenStepRandomReplacePolicy())
    if policy_name == 'replace_suff_v3':
        return ReplaceV3S(forward_fn, False, get_word_pool)
    elif policy_name == 'replace_suff_v3d':
        return ReplaceV3S(forward_fn, True, get_word_pool)
    else:
        raise ValueError


def get_erasure_v3(model_interface, discretize) -> ErasureV3:
    client = get_mmd_cache_client(model_interface, hooking_log)
    forward_fn = client.predict
    return ErasureV3(forward_fn, TenStepRandomDropPolicy(discretize))


def get_erasure_sufficiency(model_interface, discretize) -> ErasureSufficiency:
    client = get_mmd_cache_client(model_interface, hooking_log)
    forward_fn = client.predict
    return ErasureSufficiency(forward_fn, discretize)


def get_attn_v3(model_interface) -> AttnEvalMetric:
    forward_fn: AttnMaskForward = get_attn_mask_forward_fn(model_interface)
    return AttnEvalMetric(forward_fn)


def run_eval(dataset, method, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset, method)
    print("{} problems {} answers".format(len(problems), len(answers)))
    eval_policy = get_policy(model_interface, policy_name)

    scores: List[PerProblemEvalResult] = eval_v3(answers, problems, eval_policy)
    run_name = get_run_name(dataset, method, policy_name)
    save_eval_result_b(scores, run_name)


def get_policy(model_interface, policy_name) -> EvalMetricV3IF:
    if policy_name.startswith("replace_"):
        eval_policy: EvalMetricV3IF = get_replace_v3(model_interface, policy_name)
    elif policy_name == "erasure_v3":
        eval_policy: EvalMetricV3IF = get_erasure_v3(model_interface, False)
    elif policy_name == "erasure_v3d":
        eval_policy: EvalMetricV3IF = get_erasure_v3(model_interface, True)
    elif policy_name == "erasure_suff_v3":
        eval_policy: EvalMetricV3IF = get_erasure_sufficiency(model_interface, False)
    elif policy_name == "erasure_suff_v3d":
        eval_policy: EvalMetricV3IF = get_erasure_sufficiency(model_interface, True)
    elif policy_name == "attn_v3":
        eval_policy: EvalMetricV3IF = get_attn_v3(model_interface)
    else:
        raise UnexpectedPolicyException(policy_name)
    return eval_policy


def main():
    dataset = sys.argv[1]
    method = sys.argv[2]
    policy_name = sys.argv[3]
    if len(sys.argv) > 4:
        model_interface = sys.argv[4]
    else:
        model_interface = "localhost"

    run_eval(dataset, method, policy_name, model_interface)


if __name__ == "__main__":
    main()
