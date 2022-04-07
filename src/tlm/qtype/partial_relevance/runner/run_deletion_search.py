from typing import List, Callable

import numpy as np
from tlm.qtype.partial_relevance.runner.run_partial_related_eval import load_answer

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from tlm.qtype.partial_relevance.attention_based.mmd_z_client import get_mmd_client_wrap
from tlm.qtype.partial_relevance.complement_path_data_helper import load_complements
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.eval_by_erasure import EvalMetricByErasure
from tlm.qtype.partial_relevance.eval_utils import partial_related_eval
from tlm.qtype.partial_relevance.loader import load_dev_small_problems


def main():
    problems: List[RelatedEvalInstance] = load_dev_small_problems()
    answers = load_answer()
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client_wrap()
    eval_policy = EvalMetricByErasure(forward_fn,
                                      FuncContentSegJoinPolicy(),
                                      preserve_seg_idx=1,
                                      drop_rate=0.2)
    complements = load_complements()
    rewards = partial_related_eval(answers, problems, complements, eval_policy)
    print(rewards)
    print(np.log(rewards))


if __name__ == "__main__":
    main()