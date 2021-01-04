import json
import os
from typing import List, Dict

from arg.perspectives.eval_caches import get_extended_eval_candidate_as_qck
from arg.perspectives.load import get_claim_perspective_id_dict, load_train_claim_ids, get_claims_from_ids
from arg.perspectives.ppnc.resource import load_qk_candidate_train
from arg.qck.decl import QCKCandidate, QCKQuery, QKUnit, KnowledgeDocumentPart
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator
from cpath import output_path
from list_lib import lmap
from misc_lib import split_7_3, DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def is_correct_factory():
    gold = get_claim_perspective_id_dict()

    def is_correct(query: QCKQuery, candidate: QCKCandidate) -> int:
        pid_cluster = gold[int(query.query_id)]
        return int(any([int(candidate.id) in cluster for cluster in pid_cluster]))
    return is_correct


def qk_drop_content(qk: QKUnit) -> QKUnit:
    q, kdp = qk

    new_kdp_list = [KnowledgeDocumentPart("dummy", 0, 0, [])]
    return q, new_kdp_list


def main():
    split = "train"
    candidate_d: Dict[str, List[QCKCandidate]] = get_extended_eval_candidate_as_qck(split)
    generator = QCKInstanceGenerator(candidate_d, is_correct_factory())
    # claim ids split to train/val
    print("Loading data ....")
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)

    val_cids = {str(t['cId']) for t in val}
    qk_candidate: List[QKUnit] = load_qk_candidate_train()
    qk_candidate_val = list([qk for qk in qk_candidate if qk[0].query_id in val_cids])

    qk_dummy = lmap(qk_drop_content, qk_candidate_val)

    max_job = 162

    save_dir = os.path.join(output_path, "pc_qck_baseline")
    save_path = os.path.join(save_dir, "tfrecord")
    info_path = os.path.join(save_dir, "info.json")
    exist_or_mkdir(save_dir)
    data_id_manager = DataIDManager(0, 10000 * 10000)

    insts = []
    for job_id in range(max_job+1):
        todo = qk_dummy[job_id:job_id + 1]
        insts.extend(generator.generate(todo, data_id_manager))
    write_records_w_encode_fn(save_path, generator.encode_fn, insts)
    json.dump(data_id_manager.id_to_info, open(info_path, "w"))


if __name__ == "__main__":
    main()
