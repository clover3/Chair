import sys
from typing import List, Tuple

from alignment.data_structure.eval_data_structure import RelatedBinaryAnswer
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_score_dp_helper import save_eval_result_b_single
from tlm.qtype.partial_relevance.eval_utils import align_eval_b
from tlm.qtype.partial_relevance.get_policy_util import get_binary_eval_policy
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_binary_related_eval_answer


def run_align_eval_b(dataset, method, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset, method)
    print("{} problems {} answers".format(len(problems), len(answers)))
    eval_policy = get_binary_eval_policy(policy_name, model_interface, 1)
    scores: List[Tuple[str, float]] = align_eval_b(answers, problems, eval_policy)
    run_name = "{}_{}_{}".format(dataset, method, policy_name)
    save_eval_result_b_single(scores, run_name)


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
