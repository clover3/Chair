from typing import List, Tuple, Optional

from list_lib import flatten, right
from alignment.data_structure.eval_data_structure import RelatedBinaryAnswer
from tlm.qtype.partial_relevance.eval_score_dp_helper import load_eval_result_b_all
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_binary_related_eval_answer
from tlm.qtype.partial_relevance.result_print.method_preference_count import count_paired_comparison


def get_score_for_method(dataset, method, metric) -> List[float]:
    run_name = "{}_{}_{}".format(dataset, method, metric)
    eval_res: List[Tuple[str, List[Optional[float]]]] = load_eval_result_b_all(run_name)
    return list(flatten(right(eval_res)))


def get_prediction_for_method(dataset, method) -> List[List[int]]:
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset, method)
    return list(flatten(right(answers)))


all_func = get_prediction_for_method, get_score_for_method


def get_one_seg_tools(target_idx):
    def get_prediction_for_method_one_seg(dataset, method) -> List[List[int]]:
        answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset, method)
        return [a.score_table[target_idx] for a in answers]

    def get_score_for_method_one_seg(dataset, method, metric):
        run_name = "{}_{}_{}".format(dataset, method, metric)
        eval_res: List[Tuple[str, List[Optional[float]]]] = load_eval_result_b_all(run_name)
        return [seg_scores[target_idx] for seg_scores in right(eval_res)]

    return get_prediction_for_method_one_seg, get_score_for_method_one_seg


def count_cases():
    n_scores = get_score_for_method("dev_sw", "exact_match", "attn_v3")
    print("{} scores".format(len(n_scores)))


def main2():
    dataset_list = ["dev_sw"]
    # dataset_list = ["dev"]
    # method_list = ["random", "gradient", "attn_perturbation"]
    method_list = ["exact_match", "random"]
    # method_list = ["exact_match", "exact_match_noise0.1"]
    metric_list = [
        # "attn_v3",
                   "erasure_v3d", "replace_v3d",
                   "erasure_suff_v3d", "replace_suff_v3d",
                   "erasure_v3", "replace_v3",
                   "erasure_suff_v3", "replace_suff_v3",
                   ]

    get_prediction_for_method, get_score_for_method = all_func
    count_paired_comparison(dataset_list, method_list, metric_list,
                            get_score_for_method,
                            get_prediction_for_method)


if __name__ == "__main__":
    main2()
