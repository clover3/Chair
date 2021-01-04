import json
from typing import Iterable, Dict, List

from arg.perspectives.qck.qcknc_datagen import is_correct_factory
from arg.qck.decl import QCKQuery, QCKCandidate
from arg.qck.instance_generator.qc_datagen import QCInstanceGenerator
from misc_lib import DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn


def make_pc_qc(queries: Iterable[QCKQuery],
               eval_candidate: Dict[str, List[QCKCandidate]],
               save_path: str):
    generator = QCInstanceGenerator(eval_candidate, is_correct_factory())
    data_id_manager = DataIDManager(0, 10000*10000)
    insts = generator.generate(queries, data_id_manager)
    insts = list(insts)
    write_records_w_encode_fn(save_path, generator.encode_fn, insts)
    json.dump(data_id_manager.id_to_info, open(save_path + ".info", "w"))