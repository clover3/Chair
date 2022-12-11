import json
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


def get_prediction_save_path(run_name, split):
    save_path = path_join(output_path, "alamri_annotation1", "cont_classification",
                          "predictions",
                          f"{run_name}_{split}.jsonl")
    return save_path


def get_raw_prediction_save_path(run_name, split):
    save_path = path_join(output_path, "alamri_annotation1", "cont_classification",
                          "raw_predictions",
                          f"{run_name}_{split}.jsonl")
    return save_path


def save_raw_predictions(run_name, split, predictions):
    save_path = get_raw_prediction_save_path(run_name, split)
    json.dump(predictions, open(save_path, "w"))


def save_predictions(run_name, split, predictions):
    save_path = get_prediction_save_path(run_name, split)
    json.dump(predictions, open(save_path, "w"))


def load_predictions(run_name, split) -> List[int]:
    save_path = get_prediction_save_path(run_name, split)
    predictions: List[int] = json.load(open(save_path, "r"))
    return predictions


def load_problem_notes(split):
    return json.load(open(get_problem_note_path(split), "r"))