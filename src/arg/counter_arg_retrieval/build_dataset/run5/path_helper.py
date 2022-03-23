import json
import os
from typing import List, Dict, Tuple

from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import load_tsv_query
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cache import load_list_from_jsonl, StrItem
from cpath import output_path
from misc_lib import exist_or_mkdir
from trec.trec_parse import load_ranked_list_grouped


def get_swtt_dir():
    save_dir = os.path.join(output_path, "ca_building", "run5", "parsed_doc_swtt")
    exist_or_mkdir(save_dir)
    return save_dir


def get_run5_dir():
    return os.path.join(output_path, "ca_building", "run5")



def get_swtt_path(job_no):
    save_dir = get_swtt_dir()
    save_path = os.path.join(save_dir, "{}.jsonl".format(job_no))
    return save_path


def get_swtt_passage_path(query_id):
    save_dir = os.path.join(output_path, "ca_building", "run5", "swtt_passages")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, "{}.jsonl".format(query_id))
    return save_path


def get_swtt_per_query_path(query_id):
    save_dir = os.path.join(output_path, "ca_building", "run5", "per_query_doc_swtt")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, "{}.jsonl".format(query_id))
    return save_path


def get_ranked_list_per_query_path(query_id):
    save_dir = os.path.join(output_path, "ca_building", "run5", "rl_per_query", query_id)
    return save_dir


def load_swtt_jsonl_per_job(job_no) -> Dict[str, SegmentwiseTokenizedText]:
    f = open(get_swtt_path(job_no), "r")
    j_obj = json.load(f)
    out_d = {}
    for k, v in j_obj.items():
        out_d[k] = SegmentwiseTokenizedText.from_json(v)
    return out_d


def load_swtt_jsonl_per_query(query_id) -> List[StrItem[SegmentwiseTokenizedText]]:
    return load_str_swtt_pair_list(get_swtt_per_query_path(query_id))


def load_swtt_jsonl_per_query_as_d(query_id) -> Dict[str, SegmentwiseTokenizedText]:
    return {e.s: e.item for e in load_swtt_jsonl_per_query(query_id)}


def load_str_swtt_pair_list(save_path) -> List[StrItem[SegmentwiseTokenizedText]]:
    def from_json(j) -> StrItem[SegmentwiseTokenizedText]:
        return StrItem.from_json(j, SegmentwiseTokenizedText.from_json)

    return load_list_from_jsonl(save_path, from_json)


def load_qids() -> List[str]:
    qid_path = os.path.join(output_path, "ca_building", "run5", "qids.txt")
    return [line.strip() for line in open(qid_path, "r") if line.strip()]


def load_raw_rlg():
    rlg_path = os.path.join(output_path, "ca_building", "run5", "q_res.txt")
    rlg = load_ranked_list_grouped(rlg_path)
    return rlg


def load_premise_queries() -> List[Tuple[str, str]]:
    query_path = get_premise_query_path()
    return load_tsv_query(query_path)


def get_premise_query_path():
    return os.path.join(output_path, "ca_building", "run5", "queries", "premise_query.tsv")


def load_manual_queries() -> List[Tuple[str, str]]:
    query_path = os.path.join(output_path, "ca_building", "run5", "queries", "manual_query.tsv")
    return load_tsv_query(query_path)


def get_passage_prediction_path(run_name, query_id):
    save_dir = os.path.join(output_path, "ca_building", "run5", "passage_predictions")
    exist_or_mkdir(save_dir)
    save_dir = os.path.join(output_path, "ca_building", "run5", "passage_predictions", run_name)
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, "{}".format(query_id))
    return save_path


def get_ranked_list_path_to_annotate(run_name):
    return os.path.join(get_run5_dir(), "ranked_list_to_annotate", "{}.txt".format(run_name))