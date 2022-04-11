import json
import os
from typing import List

from cache import load_list_from_jsonl, save_list_to_jsonl, save_list_to_jsonl_w_fn
from contradiction.alignment.data_structure.eval_data_structure import RelatedEvalAnswer, PerProblemEvalResult
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance
from cpath import output_path, pjoin, FilePath
from misc_lib import exist_or_mkdir
from tlm.qtype.partial_relevance.eval_score_dp_helper import get_eval_score_save_path_b_all


def get_common_dir() -> FilePath:
    return pjoin(output_path, "align_nli")


def get_rei_file_path(rei_file_name):
    return os.path.join(get_common_dir(), rei_file_name)


def load_mnli_rei_problem(split):
    save_path = get_rei_file_path(f"mnli_align_{split}.jsonl")
    items: List[RelatedEvalInstance] = load_list_from_jsonl(save_path, RelatedEvalInstance.from_json)
    return items



def get_related_save_path(dataset_name, method) -> FilePath:
    file_name = f"MNLIE_{dataset_name}_{method}.score"
    save_dir = pjoin(get_common_dir(), "related_scores")
    exist_or_mkdir(save_dir)
    return pjoin(save_dir, file_name)


def save_related_eval_answer(answers: List[RelatedEvalAnswer], dataset_name, method):
    save_path = get_related_save_path(dataset_name, method)
    json.dump(answers, open(save_path, "w"), indent=True)



def save_eval_result_b(eval_res: List[PerProblemEvalResult], run_name):
    save_path = get_eval_score_save_path_b_all(run_name)
    save_list_to_jsonl_w_fn(eval_res, save_path, PerProblemEvalResult.to_json)


