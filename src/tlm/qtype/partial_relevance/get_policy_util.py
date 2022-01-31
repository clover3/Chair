from typing import Callable, List

from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.dt_deletion import DTDeletion, EvalMetricLeaveOneWC, \
    EvalMetricByErasureNoSegWC
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricWCIF, EvalMetricIF
from tlm.qtype.partial_relevance.eval_metric.eval_by_attn import EvalMetricByAttentionDrop, get_attn_mask_forward_fn, \
    EvalMetricByAttentionDropWC
from tlm.qtype.partial_relevance.eval_metric.eval_by_erasure import EvalMetricByErasure
from tlm.qtype.partial_relevance.eval_metric.partial_relevant import EvalMetricPartialRelevant, \
    EvalMetricPartialRelevant2
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import DocModFunc, get_drop_non_zero, get_drop_zero
from tlm.qtype.partial_relevance.runner.run_eval.run_partial_related_full_eval import get_mmd_client


def get_eval_policy_wc(policy_name, model_interface, preserve_idx) -> EvalMetricWCIF:
    fn: DocModFunc = get_drop_non_zero()
    if policy_name == "erasure":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricWCIF = EvalMetricByErasure(forward_fn,
                                          FuncContentSegJoinPolicy(),
                                          preserve_seg_idx=preserve_idx,
                                          doc_modify_fn=fn)
    elif policy_name == "leave_one":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        fn_drop_zero: DocModFunc = get_drop_zero()
        eval_policy: EvalMetricWCIF = EvalMetricLeaveOneWC(forward_fn,
                                           FuncContentSegJoinPolicy(),
                                           target_seg_idx=preserve_idx,
                                           doc_modify_fn=fn_drop_zero)

    elif policy_name == "partial_relevant":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricWCIF = EvalMetricPartialRelevant(forward_fn,
                                                FuncContentSegJoinPolicy(),
                                                preserve_seg_idx=preserve_idx,
                                                doc_modify_fn=fn
                                                )
    elif policy_name == "partial_relevant2":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricWCIF = EvalMetricPartialRelevant2(forward_fn,
                                                FuncContentSegJoinPolicy(),
                                                target_seg_idx=preserve_idx,
                                                doc_modify_fn=fn
                                                )
    elif policy_name == "erasure_no_seg":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricWCIF = EvalMetricByErasureNoSegWC(forward_fn,
                                               FuncContentSegJoinPolicy(),
                                               target_seg_idx=preserve_idx,
                                               doc_modify_fn=fn)
    elif policy_name == "attn":
        drop_k = 0.99
        eval_policy = EvalMetricByAttentionDropWC(get_attn_mask_forward_fn(),
                                                drop_k,
                                                preserve_seg_idx=preserve_idx,
                                                )
    else:
        raise ValueError()
    return eval_policy


def get_eval_policy(policy_name, model_interface, preserve_idx) -> EvalMetricIF:
    fn: DocModFunc = get_drop_non_zero()
    if policy_name == "erasure_no_seg":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricIF = DTDeletion(forward_fn,
                                             FuncContentSegJoinPolicy(),
                                             target_seg_idx=preserve_idx,
                                             doc_modify_fn=fn)
    elif policy_name == "attn":
        drop_k = 0.99
        eval_policy = EvalMetricByAttentionDrop(get_attn_mask_forward_fn(),
                                                drop_k,
                                                preserve_seg_idx=preserve_idx,
                                                )
    else:
        raise ValueError()
    return eval_policy