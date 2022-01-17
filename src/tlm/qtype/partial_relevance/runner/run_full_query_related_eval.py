import json
import os
from typing import List, Callable

from cpath import output_path
from tlm.qtype.partial_relevance.attention_based.mmd_z_client import get_mmd_client_wrap
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, SegmentedInstance
from tlm.qtype.partial_relevance.eval_utils import parse_related_eval_answer_from_json, related_eval
from tlm.qtype.partial_relevance.loader import load_dev_problems


# Runs eval for Related against full query
def main():
    problems: List[RelatedEvalInstance] = load_dev_problems()
    score_path = os.path.join(output_path, "qtype", "related_scores", "MMDE_dev_mmd_Z.score")
    raw_json = json.load(open(score_path, "r"))
    answers = parse_related_eval_answer_from_json(raw_json)
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client_wrap()
    drop_rate = 0.2
    rewards = related_eval(answers, problems, forward_fn, drop_rate)
    print(rewards)


if __name__ == "__main__":
    main()
