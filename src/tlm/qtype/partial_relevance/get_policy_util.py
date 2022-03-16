from typing import Callable, List

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from tlm.qtype.partial_relevance.bert_mask_interface.mmd_z_mask_cacche import get_attn_mask_forward_fn
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricIF, EvalMetricBinaryIF, \
    EvalMetricBinaryWCIF
from tlm.qtype.partial_relevance.eval_metric.eval_by_attn import EvalMetricByAttentionDrop, \
    EvalMetricByAttentionAUC, EvalMetricByAttentionBrevity, EvalMetricByAttentionDropB
from tlm.qtype.partial_relevance.eval_metric.eval_by_erasure import EvalMetricByErasure
from tlm.qtype.partial_relevance.eval_metric.partial_relevant import EvalMetricPartialRelevant, \
    EvalMetricPartialRelevant2
from tlm.qtype.partial_relevance.eval_metric.ps_replace_helper import get_ps_replace_100words, get_ps_replace_empty
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_drop_non_zero, get_drop_zero, \
    DocModFuncB
from tlm.qtype.partial_relevance.eval_metric_binary.dt_deletion import DTDeletion, EvalMetricLeaveOneWC, \
    EvalMetricByErasureNoSegWC
from tlm.qtype.partial_relevance.runner.run_eval_old.run_partial_related_full_eval import get_mmd_client


def get_binary_eval_policy_wc(policy_name, model_interface, preserve_idx) -> EvalMetricBinaryWCIF:
    fn: DocModFuncB = get_drop_non_zero()
    if policy_name == "erasure":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricBinaryWCIF = EvalMetricByErasure(forward_fn,
                                                                FuncContentSegJoinPolicy(),
                                                                preserve_seg_idx=preserve_idx,
                                                                doc_modify_fn=fn)
    elif policy_name == "leave_one":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        fn_drop_zero: DocModFuncB = get_drop_zero()
        eval_policy: EvalMetricBinaryWCIF = EvalMetricLeaveOneWC(forward_fn,
                                                                 FuncContentSegJoinPolicy(),
                                                                 target_seg_idx=preserve_idx,
                                                                 doc_modify_fn=fn_drop_zero)

    elif policy_name == "partial_relevant":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricBinaryWCIF = EvalMetricPartialRelevant(forward_fn,
                                                                      FuncContentSegJoinPolicy(),
                                                                      preserve_seg_idx=preserve_idx,
                                                                      doc_modify_fn=fn
                                                                      )
    elif policy_name == "partial_relevant2":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricBinaryWCIF = EvalMetricPartialRelevant2(forward_fn,
                                                                       FuncContentSegJoinPolicy(),
                                                                       target_seg_idx=preserve_idx,
                                                                       doc_modify_fn=fn
                                                                       )
    elif policy_name == "erasure_no_seg":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricBinaryWCIF = EvalMetricByErasureNoSegWC(forward_fn,
                                                                       FuncContentSegJoinPolicy(),
                                                                       target_seg_idx=preserve_idx,
                                                                       doc_modify_fn=fn)
    else:
        raise ValueError()
    return eval_policy


def get_binary_eval_policy(policy_name, model_interface, target_seg_idx) -> EvalMetricBinaryIF:
    if policy_name == "deletion":
        fn: DocModFuncB = get_drop_non_zero()
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy: EvalMetricBinaryIF = DTDeletion(forward_fn,
                                               FuncContentSegJoinPolicy(),
                                               target_seg_idx=target_seg_idx,
                                               doc_modify_fn=fn)
    elif policy_name == "ps_replace_precision":
        eval_policy: EvalMetricBinaryIF = get_ps_replace_100words(model_interface, target_seg_idx, "precision")
    elif policy_name == "ps_replace_recall":
        eval_policy: EvalMetricBinaryIF = get_ps_replace_100words(model_interface, target_seg_idx, "recall")
    elif policy_name == "ps_deletion_precision":
        eval_policy: EvalMetricBinaryIF = get_ps_replace_empty(model_interface, target_seg_idx, "precision")
    elif policy_name == "ps_deletion_recall":
        eval_policy: EvalMetricBinaryIF = get_ps_replace_empty(model_interface, target_seg_idx, "recall")
    elif policy_name == "attn":
        eval_policy: EvalMetricBinaryIF = EvalMetricByAttentionDropB(get_attn_mask_forward_fn(model_interface),
                                                                     preserve_seg_idx=target_seg_idx)
    else:
        raise ValueError()
    return eval_policy


def get_real_val_eval_policy(policy_name, model_interface, target_seg_idx) -> EvalMetricIF:
    if policy_name == "attn":
        drop_k = 0.25
        eval_policy = EvalMetricByAttentionDrop(get_attn_mask_forward_fn(model_interface),
                                                drop_k,
                                                preserve_seg_idx=target_seg_idx,
                                                )
    elif policy_name == "attn_auc":
        eval_policy = EvalMetricByAttentionAUC(get_attn_mask_forward_fn(model_interface),
                                               target_seg_idx=target_seg_idx,
                                               )
    elif policy_name == "attn_brevity":
        eval_policy = EvalMetricByAttentionBrevity(get_attn_mask_forward_fn(model_interface),
                                               target_seg_idx=target_seg_idx,
                                               )
    else:
        raise ValueError()
    return eval_policy

