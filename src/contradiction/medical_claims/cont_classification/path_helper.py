from typing import List, Iterable, Callable, Dict, Tuple, Set
from cache import load_list_from_jsonl
from contradiction.medical_claims.cont_classification.defs import ContProblem
from cpath import output_path
from misc_lib import path_join


def get_problem_path(split):
    save_path = path_join(output_path, "alamri_annotation1", "cont_classification", split + ".jsonl")
    return save_path


def get_problem_note_path(split):
    save_path = path_join(output_path, "alamri_annotation1", "cont_classification", split + "_note.jsonl")
    return save_path


def load_cont_classification_problems(split) -> List[ContProblem]:
    save_path = get_problem_path(split)
    return load_list_from_jsonl(save_path, ContProblem.from_json)
