from typing import List, Callable

import numpy as np

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from tlm.qtype.partial_relevance.attention_based.mmd_z_client import get_mmd_client_wrap
from tlm.qtype.partial_relevance.complement_path_data_helper import load_complements
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.eval_by_erasure import EvalMetricByErasure
from tlm.qtype.partial_relevance.eval_metric.partial_relevant import EvalMetricPartialRelevant
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import DocModFuncR, get_top_k_fn
from tlm.qtype.partial_relevance.eval_utils import partial_related_eval
from tlm.qtype.partial_relevance.loader import load_dev_small_problems
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer


def load_answer():
    dataset_name = "dev_sm"
    method = "perturbation"
    return load_related_eval_answer(dataset_name, method)


# Runs eval for Related against full query
def main():
    problems: List[RelatedEvalInstance] = load_dev_small_problems()
    answers = load_answer()
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client_wrap()
    fn = get_top_02_drop_fn()
    eval_policy = EvalMetricPartialRelevant(forward_fn,
                                            FuncContentSegJoinPolicy(),
                                            preserve_seg_idx=1,
                                            doc_modify_fn=fn
                                            )
    complements = load_complements()
    rewards = partial_related_eval(answers, problems, complements, eval_policy)
    print(rewards)


def main2():
    problems: List[RelatedEvalInstance] = load_dev_small_problems()
    answers = load_answer()
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client_wrap()
    fn = get_top_02_drop_fn()
    eval_policy = EvalMetricByErasure(forward_fn,
                                      FuncContentSegJoinPolicy(),
                                      preserve_seg_idx=1,
                                      doc_modify_fn=fn)
    complements = load_complements()
    rewards = partial_related_eval(answers, problems, complements, eval_policy)
    print(rewards)
    print(np.log(rewards))


def get_top_02_drop_fn():
    fn: DocModFuncR = get_top_k_fn(0.2)
    return fn


if __name__ == "__main__":
    main2()
