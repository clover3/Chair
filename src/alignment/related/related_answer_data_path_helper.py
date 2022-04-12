import json
import os
from typing import List

from alignment.data_structure.ds_helper import parse_related_eval_answer_from_json, \
    parse_related_binary_answer_from_json
from cpath import output_path
from misc_lib import exist_or_mkdir
from alignment.data_structure.eval_data_structure import RelatedEvalAnswer, RelatedBinaryAnswer


def get_related_save_path(dataset_name, method):
    save_path = os.path.join(output_path, "align_nli", "related_scores", "MNLIE_{}_{}.score".format(dataset_name, method))
    return save_path


def save_related_eval_answer(answers: List[RelatedEvalAnswer], dataset_name, method):
    save_path = get_related_save_path(dataset_name, method)
    json.dump(answers, open(save_path, "w"), indent=True)


def load_related_answer(run_name) -> List[RelatedEvalAnswer]:
    score_path = os.path.join(output_path, "align_nli", "related_scores", "{}.score".format(run_name))
    raw_json = json.load(open(score_path, "r"))
    return parse_related_eval_answer_from_json(raw_json)


def load_related_eval_answer(dataset_name, method) -> List[RelatedEvalAnswer]:
    score_path = get_related_save_path(dataset_name, method)
    raw_json = json.load(open(score_path, "r"))
    return parse_related_eval_answer_from_json(raw_json)


def get_related_binary_save_path(dataset_name, method):
    dir_path = os.path.join(output_path, "align_nli", "binary_related_scores")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, "MNLIE_{}_{}.score".format(dataset_name, method))
    return save_path


def save_binary_related_eval_answer(answers: List[RelatedBinaryAnswer], dataset_name, method):
    save_path = get_related_binary_save_path(dataset_name, method)
    save_json_at(answers, save_path)


def save_json_at(data, save_path):
    json.dump(data, open(save_path, "w"), indent=True)


def load_binary_related_eval_answer(dataset_name, method) -> List[RelatedBinaryAnswer]:
    score_path = get_related_binary_save_path(dataset_name, method)
    raw_json = json.load(open(score_path, "r"))
    return parse_related_binary_answer_from_json(raw_json)

