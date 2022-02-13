from typing import List, Tuple

from tlm.qtype.partial_relevance.calc_avg import load_eval_result_r
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_score_dp_helper import save_eval_result_r
from tlm.qtype.partial_relevance.loader import load_mmde_problem


# Runs eval for Related against full query
def main():
    dataset = "dev"
    method_list = ["random", "gradient", "attn_perturbation"]
    metric_list = ["partial_relevant", "erasure"]
    split_pos_neg(dataset, method_list, metric_list)


# Runs eval for Related against full query
def main2():
    dataset = "dev_word"
    # method_list = ["random", "gradient", "attn_perturbation"]
    method_list = ["exact_match"]
    metric_list = ["partial_relevant", "erasure"]
    split_pos_neg(dataset, method_list, metric_list)


def split_pos_neg(dataset, method_list, metric_list):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)

    def is_pos(s):
        return s >= 0.5

    pos_problem_ids = [p.problem_id for p in problems if is_pos(p.score)]
    neg_problem_ids = [p.problem_id for p in problems if not is_pos(p.score)]

    def split_save(dataset, method, policy_name):
        run_name = "{}_{}_{}".format(dataset, method, policy_name)
        eval_res: List[Tuple[str, float]] = load_eval_result_r(run_name)
        pos_res = [(problem_id, score) for problem_id, score in eval_res if problem_id in pos_problem_ids]
        neg_res = [(problem_id, score) for problem_id, score in eval_res if problem_id in neg_problem_ids]
        assert len(pos_res) + len(neg_res) == len(eval_res)
        new_run_name_p = "{}p_{}_{}".format(dataset, method, policy_name)
        new_run_name_n = "{}n_{}_{}".format(dataset, method, policy_name)
        save_eval_result_r(pos_res, new_run_name_p)
        save_eval_result_r(neg_res, new_run_name_n)

    for method in method_list:
        for metric in metric_list:
            split_save(dataset, method, metric)


if __name__ == "__main__":
    main2()
