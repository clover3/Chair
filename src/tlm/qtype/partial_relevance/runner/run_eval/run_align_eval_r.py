import sys
from typing import List, Tuple

from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_score_dp_helper import save_eval_result_r
from tlm.qtype.partial_relevance.eval_utils import align_eval_r
from tlm.qtype.partial_relevance.get_policy_util import get_real_val_eval_policy
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer


def run_eval(dataset, method, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers = load_related_eval_answer(dataset, method)
    print("{} problems {} answers".format(len(problems), len(answers)))
    eval_policy = get_real_val_eval_policy(policy_name, model_interface, 1)
    scores: List[Tuple[str, float]] = align_eval_r(answers, problems, eval_policy)
    run_name = "{}_{}_{}".format(dataset, method, policy_name)
    save_eval_result_r(scores, run_name)


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
