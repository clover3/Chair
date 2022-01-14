import json
import os
from typing import List, Callable, Dict

import numpy as np

from cpath import output_path
from list_lib import index_by_fn
from tlm.qtype.partial_relevance.attention_based.mmd_z_client import get_mmd_client_wrap
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, SegmentedInstance, \
    ContributionSummary
from tlm.qtype.partial_relevance.loader import load_dev_small_problems
from trainer.promise import MyPromise, PromiseKeeper


def parse_related_eval_answer_from_json(raw_json) -> List[RelatedEvalAnswer]:
    def parse_entry(e) -> RelatedEvalAnswer:
        problem_id = e[0]
        score_array_wrap = e[1]
        score_array = score_array_wrap[0]
        assert type(score_array[0][0]) == float
        return RelatedEvalAnswer(problem_id, ContributionSummary(score_array))
    return list(map(parse_entry, raw_json))


def load_eval_answer(run_name) -> List[RelatedEvalAnswer]:
    score_path = os.path.join(output_path, "qtype", "related_scores", "{}.score".format(run_name))
    raw_json = json.load(open(score_path, "r"))
    return parse_related_eval_answer_from_json(raw_json)


#  candidate metric
#   1. Area Under Reserve/Erasure Curve
#   2. Top-k drop flip.


def get_top_k_drop(a: RelatedEvalAnswer, si: RelatedEvalInstance, k, target_seg1_idx):
    seg2_len = si.seg_instance.get_seg2_len()
    drop_len = int(seg2_len * k)
    print(target_seg1_idx)
    scores: List[float] = a.contribution.table[target_seg1_idx]
    argsorted = np.argsort(scores)
    drop_indices = argsorted[-drop_len:]  # Pick one with highest scores
    new_seg_instance = si.seg_instance.get_seg2_dropped_instances(drop_indices)
    return new_seg_instance


def do_eval(answer_list: List[RelatedEvalAnswer],
            problem_list: List[RelatedEvalInstance],
            forward_fn: Callable[[List[SegmentedInstance]], List[float]]
            ) -> List[float]:
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problem_list)
    k = 0.2
    pk = PromiseKeeper(forward_fn)
    payloads = []
    for answer in answer_list:
        problem: RelatedEvalInstance = pid_to_p[answer.problem_id]

        future_prediction_list = []
        for seg_idx in range(problem.seg_instance.get_seg1_len()):
            new_seg = get_top_k_drop(answer, problem, k, seg_idx)
            future_prediction = MyPromise(new_seg, pk).future()
            future_prediction_list.append(future_prediction)
        payloads.append((problem, future_prediction_list))

    pk.do_duty(True)

    eval_score_list = []
    for problem, future_prediction_list in payloads:
        for idx, future_prediction in enumerate(future_prediction_list):
            new_prediction = future_prediction.get()
            original_prediction = problem.seg_instance.score
            eval_score = original_prediction - new_prediction  # Higher the better
            print("{0:.2f} -> {1:.2f}".format(original_prediction, new_prediction))
            eval_score_list.append(eval_score)
    return eval_score_list


def main():
    problems: List[RelatedEvalInstance] = load_dev_small_problems()
    score_path = os.path.join(output_path, "qtype", "related_scores", "MMDE_dev_mmd_Z.score")
    raw_json = json.load(open(score_path, "r"))
    answers = parse_related_eval_answer_from_json(raw_json)
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client_wrap()
    rewards = do_eval(answers, problems, forward_fn)
    print(rewards)


if __name__ == "__main__":
    main()
