import json
import os
from typing import List

from cpath import output_path
from misc_lib import exist_or_mkdir
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalAnswer, ContributionSummary, RelatedBinaryAnswer


def get_related_save_path(dataset_name, method):
    save_path = os.path.join(output_path, "qtype", "related_scores", "MMDE_{}_{}.score".format(dataset_name, method))
    return save_path


def save_related_eval_answer(answers: List[RelatedEvalAnswer], dataset_name, method):
    save_path = get_related_save_path(dataset_name, method)
    json.dump(answers, open(save_path, "w"), indent=True)


def parse_related_eval_answer_from_json(raw_json) -> List[RelatedEvalAnswer]:
    def parse_entry(e) -> RelatedEvalAnswer:
        problem_id = e[0]
        score_array_wrap = e[1]
        score_array = score_array_wrap[0]
        score_type = type(score_array[0][0])
        assert score_type == float or score_type == int
        return RelatedEvalAnswer(problem_id, ContributionSummary(score_array))
    return list(map(parse_entry, raw_json))


def load_related_answer(run_name) -> List[RelatedEvalAnswer]:
    score_path = os.path.join(output_path, "qtype", "related_scores", "{}.score".format(run_name))
    raw_json = json.load(open(score_path, "r"))
    return parse_related_eval_answer_from_json(raw_json)


def load_related_eval_answer(dataset_name, method) -> List[RelatedEvalAnswer]:
    score_path = get_related_save_path(dataset_name, method)
    raw_json = json.load(open(score_path, "r"))
    return parse_related_eval_answer_from_json(raw_json)


def get_related_binary_save_path(dataset_name, method, policy):
    dir_path = os.path.join(output_path, "qtype", "binary_related_scores")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, "MMDE_{}_{}_{}.score".format(dataset_name, method, policy))
    return save_path


def save_binary_related_eval_answer(answers: List[RelatedBinaryAnswer], dataset_name, method, policy):
    save_path = get_related_binary_save_path(dataset_name, method, policy)
    json.dump(answers, open(save_path, "w"), indent=True)


def parse_related_binary_answer_from_json(raw_json) -> List[RelatedBinaryAnswer]:
    def parse_entry(e) -> RelatedBinaryAnswer:
        problem_id = e[0]
        indices_list = e[1]
        assert type(indices_list[0][0]) == int
        return RelatedBinaryAnswer(problem_id, indices_list)
    return list(map(parse_entry, raw_json))


def load_binary_related_eval_answer(dataset_name, method, policy) -> List[RelatedBinaryAnswer]:
    score_path = get_related_binary_save_path(dataset_name, method, policy)
    raw_json = json.load(open(score_path, "r"))
    return parse_related_binary_answer_from_json(raw_json)

