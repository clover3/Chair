import json
import os

from arg.perspectives.load import splits
from arg.perspectives.qck.qck_common import get_qck_queries
from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.qck.qc_datagen import QCInstanceGenerator
from cpath import output_path
from misc_lib import DataIDManager, exist_or_mkdir


def make_pc_qc(queries, eval_candidate, save_path):
    generator = QCInstanceGenerator(eval_candidate, is_correct_factory())
    data_id_manager = DataIDManager(0, 10000*10000)
    insts = generator.generate(queries, data_id_manager)
    insts = list(insts)
    #write_records_w_encode_fn(save_path, generator.encode_fn, insts)
    json.dump(data_id_manager.id_to_info, open(save_path + ".info", "w"))


def main():
    save_dir = os.path.join(output_path, "pc_qc")
    exist_or_mkdir(save_dir)
    for split in splits:
        queries = get_qck_queries(split)
        eval_candidate = get_eval_candidates_as_qck(split)
        save_path = os.path.join(save_dir, split)
        make_pc_qc(queries, eval_candidate, save_path)


if __name__ == "__main__":
    main()