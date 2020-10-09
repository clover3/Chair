import json
import os
import sys
from typing import List, Dict, Tuple

from arg.perspectives.eval_caches import get_eval_candidates_from_pickle
from arg.perspectives.qck.qck_common import get_qck_queries, get_qck_candidate_from_candidate_id
from arg.qck.decl import QCKQuery, QCKCandidate, qk_convert_map
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.qck.qk_summarize import collect_good_passages, QKOutEntry, qck_from_qk_results, \
    write_qck_as_tfrecord
from base_type import FilePath
from cpath import output_path
from list_lib import dict_value_map, lmap, dict_key_map, flatten
from misc_lib import exist_or_mkdir


# all dirty code that are specific to perspective dataset should be in this function
def make_qcknc_problem(passage_score_path: FilePath,
                       info_path: FilePath,
                       config_path: FilePath,
                       split: str,
                       save_name: str,
                       ) -> None:

    config = json.load(open(config_path, "r"))
    data_id_to_info: Dict = load_combine_info_jsons(info_path, qk_convert_map)

    print("number of dat info ", len(data_id_to_info))
    qk_result: List[Tuple[str, List[QKOutEntry]]] = collect_good_passages(data_id_to_info, passage_score_path, config)

    queries: List[QCKQuery] = get_qck_queries(split)
    query_dict = {q.query_id: q for q in queries}

    candidate_perspectives: Dict[int, List[Dict]] = dict(get_eval_candidates_from_pickle(split))

    def get_pids(l: List[Dict]) -> List[str]:
        return lmap(lambda x: x['pid'], l)

    candidate_id_dict_1: Dict[int, List[str]] = dict_value_map(get_pids, candidate_perspectives)
    candidate_id_dict: Dict[str, List[str]] = dict_key_map(str, candidate_id_dict_1)

    all_candidate_ids = set(flatten(candidate_id_dict.values()))
    candidate_dict: Dict[str, QCKCandidate] = {cid: get_qck_candidate_from_candidate_id(cid) for cid in all_candidate_ids}
    payloads = qck_from_qk_results(qk_result, candidate_id_dict, query_dict, candidate_dict)

    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    save_path = os.path.join(out_dir, save_name + ".tfrecord")
    data_id_man = write_qck_as_tfrecord(save_path, payloads)
    info_save_path = os.path.join(out_dir, save_name + ".info")
    print("Payload size : ", len(data_id_man.id_to_info))

    json.dump(data_id_man.id_to_info, open(info_save_path, "w"))
    print("tfrecord saved at :", save_path)
    print("info saved at :", info_save_path)


def main():
    run_config = json.load(open(sys.argv[1], "r"))
    make_qcknc_problem(run_config['prediction_path'],
                       run_config['info_path'],
                       run_config['config_path'],
                       run_config['split'],
                       run_config['save_name']
                       )


if __name__ == "__main__":
    main()

