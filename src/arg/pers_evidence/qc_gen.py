import json
import os
from typing import List, Dict

from arg.pers_evidence.common import get_qck_queries
from arg.pers_evidence.runner.get_candidate_dict import load_candidate
from arg.perspectives.load import splits, evidence_gold_dict_str_str
from arg.qck.decl import QCKQuery, QCKCandidate
from arg.qck.qc_datagen import QCInstanceGenerator
from arg.qck.qcknc_datagen import QCKCandidateI
from cpath import output_path
from misc_lib import DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def get_is_correct_fn():
    gold_dict: Dict[str, List[str]] = evidence_gold_dict_str_str()

    def is_correct(query: QCKQuery, candidate: QCKCandidate) -> bool:
        gold_e_ids = gold_dict[query.query_id]
        if candidate.id in gold_e_ids:
            return True
        else:
            return False
    return is_correct


def generate_qc_bert():
    is_correct_fn = get_is_correct_fn()
    save_dir = os.path.join(output_path, "pc_evidence_qc")
    exist_or_mkdir(save_dir)
    for split in splits:
        candidate_dict: Dict[str, List[QCKCandidateI]] = load_candidate(split)
        queries = get_qck_queries(split)
        generator = QCInstanceGenerator(candidate_dict, is_correct_fn)
        data_id_manager = DataIDManager()
        insts = generator.generate(queries, data_id_manager)

        save_path = os.path.join(save_dir, split)
        write_records_w_encode_fn(save_path, generator.encode_fn, insts)
        json.dump(data_id_manager.id_to_info, open(save_path + ".info", "w"))


if __name__ == "__main__":
    generate_qc_bert()
