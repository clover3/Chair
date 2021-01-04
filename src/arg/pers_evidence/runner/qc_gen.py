import json
import os
from typing import List, Dict

from arg.pers_evidence.common import get_qck_queries
from arg.pers_evidence.runner.get_candidate_dict import get_ex_candidate_for_training, \
    get_candidate
from arg.perspectives.load import splits, evidence_gold_dict_str_str
from arg.qck.decl import QCKQuery, QCKCandidate
from arg.qck.instance_generator.qc_datagen import QCInstanceGenerator
from arg.qck.instance_generator.qcknc_datagen import QCKCandidateI
from cpath import output_path
from misc_lib import DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def get_is_correct_fn():
    gold_dict: Dict[str, List[str]] = evidence_gold_dict_str_str()

    def is_correct(query: QCKQuery, candidate: QCKCandidate) -> bool:
        gold_e_ids: List[str] = gold_dict[query.query_id]
        if candidate.id in gold_e_ids:
            return True
        else:
            return False
    return is_correct


def do_generate_jobs(candidate_dict, is_correct_fn, save_dir, split):
    queries = get_qck_queries(split)
    generator = QCInstanceGenerator(candidate_dict, is_correct_fn)
    data_id_manager = DataIDManager()
    insts = generator.generate(queries, data_id_manager)
    save_path = os.path.join(save_dir, split)
    write_records_w_encode_fn(save_path, generator.encode_fn, insts)
    json.dump(data_id_manager.id_to_info, open(save_path + ".info", "w"))


def generate_qc3():
    is_correct_fn = get_is_correct_fn()
    save_dir = os.path.join(output_path, "pc_evidence_qc3")
    exist_or_mkdir(save_dir)
    for split in splits:
        candidate_dict: Dict[str, List[QCKCandidateI]] = get_candidate(split)
        do_generate_jobs(candidate_dict, is_correct_fn, save_dir, split)


def generate_qc_bert_bal():
    is_correct_fn = get_is_correct_fn()
    save_dir = os.path.join(output_path, "pc_evidence_qc_bal")
    exist_or_mkdir(save_dir)
    for split in splits:
        candidate_dict: Dict[str, List[QCKCandidateI]] = get_ex_candidate_for_training(split)
        do_generate_jobs(candidate_dict, is_correct_fn, save_dir, split)


def generate_qc_bert4():
    is_correct_fn = get_is_correct_fn()
    save_dir = os.path.join(output_path, "pc_evidence_qc4")
    exist_or_mkdir(save_dir)
    for split in splits:
        candidate_dict: Dict[str, List[QCKCandidateI]] = get_ex_candidate_for_training(split, False)
        do_generate_jobs(candidate_dict, is_correct_fn, save_dir, split)


if __name__ == "__main__":
    generate_qc_bert4()
