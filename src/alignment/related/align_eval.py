import numpy as np
from bert_api.segmented_instance.seg_instance import SegmentedInstance
from alignment.data_structure.eval_data_structure import RelatedBinaryAnswer, PerProblemEvalResult, \
    UnexpectedPolicyException
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from alignment.nli_align_path_helper import load_mnli_rei_problem, save_eval_result_b
from alignment.related.related_answer_data_path_helper import load_binary_related_eval_answer
from bert_api.task_clients.nli_interface.nli_interface import get_nli_cache_client, NLIInput
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_100_random_spans
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import TenStepRandomReplacePolicy, \
    TenStepRandomDropPolicy
from tlm.qtype.partial_relevance.eval_metric_binary.attn_v3 import AttnEvalMetric
from tlm.qtype.partial_relevance.eval_metric_binary.erasure_v3 import ErasureV3, ErasureSufficiency
from tlm.qtype.partial_relevance.eval_metric_binary.eval_v3 import eval_v3
from tlm.qtype.partial_relevance.eval_metric_binary.eval_v3_common import MetricV3, EvalMetricV3IF
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v3 import ReplaceV3, ReplaceV3S
from tlm.qtype.partial_relevance.eval_score_dp_helper import get_run_name
from typing import List, Callable
import scipy.special


def get_nli_ce_client(model_interface) -> Callable[[List[SegmentedInstance]], List[float]]:
    cache_client = get_nli_cache_client(model_interface)

    def predict_fn(items: List[SegmentedInstance]) -> List[float]:
        if not items:
            return []
        inputs: List[NLIInput] = [NLIInput(item.text2, item.text1) for item in items]
        logits: List[List[float]] = cache_client.predict(inputs)
        logits_np = np.array(logits)
        probs = scipy.special.softmax(logits_np, axis=1)
        return 1 - probs[:, 1]

    return predict_fn


def get_replace_v3(model_interface, policy_name) -> MetricV3:
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_nli_ce_client(model_interface)
    replacee_span_list = get_100_random_spans()

    def get_word_pool(e) -> List[List[int]]:
        words = [span.ngram_ids for span in replacee_span_list]
        return words
    if policy_name == 'replace_v3':
        return ReplaceV3(forward_fn, TenStepRandomReplacePolicy(), get_word_pool)
    elif policy_name == 'replace_v3d':
        return ReplaceV3(forward_fn, TenStepRandomReplacePolicy(True), get_word_pool)
    if policy_name == 'replace_suff_v3':
        return ReplaceV3S(forward_fn, False, get_word_pool)
    elif policy_name == 'replace_suff_v3d':
        return ReplaceV3S(forward_fn, True, get_word_pool)
    else:
        raise ValueError


def get_erasure_v3(model_interface, discretize) -> ErasureV3:
    forward_fn = get_nli_ce_client(model_interface)
    return ErasureV3(forward_fn, TenStepRandomDropPolicy(discretize))


def get_erasure_sufficiency(model_interface, discretize) -> ErasureSufficiency:
    forward_fn = get_nli_ce_client(model_interface)
    return ErasureSufficiency(forward_fn, discretize)


def get_attn_v3(model_interface) -> EvalMetricV3IF:
    forward_fn: AttnMaskForward = get_attn_mask_forward_fn(model_interface)
    return AttnEvalMetric(forward_fn)


def get_policy(model_interface, policy_name):
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


def run_eval(dataset, method, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset)
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset, method)
    print("{} problems {} answers".format(len(problems), len(answers)))
    eval_policy = get_policy(model_interface, policy_name)

    scores: List[PerProblemEvalResult] = eval_v3(answers, problems, eval_policy)
    run_name = get_run_name(dataset, method, policy_name)
    save_eval_result_b(scores, run_name)