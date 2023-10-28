import json

from dataset_specific.beir_eval.path_helper import get_json_qres_save_path


def save_json_qres(run_name: str, output):
    json_qres_save_path = get_json_qres_save_path(run_name)
    json.dump(output, open(json_qres_save_path, "w"))


def load_json_qres(run_name: str):
    json_qres_save_path = get_json_qres_save_path(run_name)
    return json.load(open(json_qres_save_path, "r"))


