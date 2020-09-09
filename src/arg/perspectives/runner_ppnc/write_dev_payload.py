import os
from typing import List

from arg.perspectives.ppnc.cpnr_predict_datagen import get_encode_fn, generate_instances
from arg.perspectives.ppnc.decl import ClaimPassages
from arg.perspectives.ppnc.kd_payload import load_dev_payload
from cache import save_to_pickle
from cpath import output_path
from misc_lib import DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn


def main():
    raw_payload: List[ClaimPassages] = load_dev_payload()
    save_path = os.path.join(output_path, "pc_dev_passage_payload")
    encode = get_encode_fn(512)
    data_id_manage = DataIDManager()
    insts = list(generate_instances(raw_payload, data_id_manage))
    write_records_w_encode_fn(save_path, encode, insts, len(insts))
    save_to_pickle(data_id_manage.id_to_info, "pc_dev_passage_payload_info")


if __name__ == "__main__":
    main()