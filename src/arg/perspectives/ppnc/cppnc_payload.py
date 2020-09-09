import json
import os
from typing import Dict, List, Tuple, Iterable

from arg.perspectives.ppnc.parse_cpnr_results import collect_good_passages, join_perspective, put_texts
from arg.perspectives.ppnc.ppnc_decl import PayloadAsTokens
from base_type import FilePath
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def make_cppnc_problem(passage_score_path: FilePath,
                       data_id_to_info: Dict,
                       claims: List[Dict],
                       candidate_perspectives,
                       config,
                       save_name: str,
                       encode_inner_fn
                       ):
    output: List[Tuple[int, List[Dict]]] = collect_good_passages(data_id_to_info, passage_score_path, config)
    joined_payloads: List = list(join_perspective(output, candidate_perspectives))
    tokenizer = get_tokenizer()
    data_id_man = DataIDManager()

    payloads: Iterable[PayloadAsTokens] = put_texts(joined_payloads, claims, tokenizer, data_id_man)
    max_seq_length = 512

    def encode_fn(r: PayloadAsTokens):
        return encode_inner_fn(max_seq_length, tokenizer, r)

    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    save_path = os.path.join(out_dir, save_name + ".tfrecord")
    write_records_w_encode_fn(save_path, encode_fn, payloads)
    info_save_path = os.path.join(out_dir, save_name + ".info")
    print("Payload size : ", len(data_id_man.id_to_info))

    json.dump(data_id_man.id_to_info, open(info_save_path, "w"))
    print("tfrecord saved at :", save_path)
    print("info saved at :", info_save_path)

