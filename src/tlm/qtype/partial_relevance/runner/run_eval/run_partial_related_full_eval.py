from typing import List, Callable, Tuple

from tlm.qtype.mmd_z_direct import get_mmd_z_direct_wrap
from tlm.qtype.partial_relevance.attention_based.mmd_z_client import get_mmd_client_wrap
from tlm.qtype.partial_relevance.complement_path_data_helper import load_complements
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.doc_modify_fns import DocModFunc, get_top_k_fn
from tlm.qtype.partial_relevance.eval_metric.eval_by_attn import EvalMetricByAttentionDrop, get_attn_mask_forward_fn
from tlm.qtype.partial_relevance.eval_metric.eval_by_erasure import EvalMetricByErasure
from tlm.qtype.partial_relevance.eval_metric.partial_relevant import EvalMetricPartialRelevant, \
    EvalMetricPartialRelevant2
from tlm.qtype.partial_relevance.eval_score_dp_helper import save_eval_result
from tlm.qtype.partial_relevance.eval_utils import partial_related_eval
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer


def get_mmd_client(option: str):
    if option == "localhost":
        return get_mmd_client_wrap()
    elif option == "direct":
        print("use direct predictor")
        return get_mmd_z_direct_wrap()
    else:
        raise ValueError


def get_eval_policy(policy_name, model_interface):
    drop_k = 0.2
    fn: DocModFunc = get_top_k_fn(drop_k)
    if policy_name == "erasure":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy = EvalMetricByErasure(forward_fn,
                                          FuncContentSegJoinPolicy(),
                                          preserve_seg_idx=1,
                                          doc_modify_fn=fn)
    elif policy_name == "partial_relevant":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy = EvalMetricPartialRelevant(forward_fn,
                                                FuncContentSegJoinPolicy(),
                                                preserve_seg_idx=1,
                                                doc_modify_fn=fn
                                                )
    elif policy_name == "partial_relevant2":
        forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)
        eval_policy = EvalMetricPartialRelevant2(forward_fn,
                                                FuncContentSegJoinPolicy(),
                                                target_seg_idx=0,
                                                doc_modify_fn=fn
                                                )
    elif policy_name == "attn":
        drop_k = 0.99
        eval_policy = EvalMetricByAttentionDrop(get_attn_mask_forward_fn(),
                                                drop_k,
                                                preserve_seg_idx=1,
                                                )
    else:
        raise ValueError()
    return eval_policy


def run_eval(dataset, method, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers = load_related_eval_answer(dataset, method)
    print("{} problems {} answers".format(len(problems), len(answers)))
    eval_policy = get_eval_policy(policy_name, model_interface)
    complements = load_complements()
    scores: List[Tuple[str, float]] = partial_related_eval(answers, problems, complements, eval_policy)
    run_name = "{}_{}_{}".format(dataset, method, policy_name)
    save_eval_result(scores, run_name)


# Runs eval for Related against full query
def main_partial_relevant():
    dataset = "dev"
    method = "gradient"
    method = "attn_perturbation"
    policy_name = "partial_relevant"
    run_eval(dataset, method, policy_name)


# Runs eval for Related against full query
def main_erasure():
    dataset = "dev"
    method = "gradient"
    policy_name = "erasure"
    run_eval(dataset, method, policy_name)


# Runs eval for Related against full query
def main_attn():
    dataset = "dev_sm"
    method = "gradient"
    policy_name = "attn"
    run_eval(dataset, method, policy_name)


if __name__ == "__main__":
    main_attn()
