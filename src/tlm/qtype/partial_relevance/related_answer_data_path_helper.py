import json
import os
from typing import List

from cpath import output_path
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalAnswer, ContributionSummary


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
        assert type(score_array[0][0]) == float
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


