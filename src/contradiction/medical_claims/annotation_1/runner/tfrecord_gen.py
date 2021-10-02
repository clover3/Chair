import json
import os
from typing import List, Callable, Tuple, Iterator

from contradiction.medical_claims.annotation_1.load_data import load_alamri1_all
from cpath import output_path
from data_generator.cls_sep_encoder import get_text_pair_encode_fn, PairedInstance
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def generate_and_write(file_name,
                       generate_fn: Callable[[DataIDManager], Iterator[PairedInstance]],
                       tokenizer):
    max_seq_length = 300
    save_dir = os.path.join(output_path, "alamri_annotation1", "tfrecord")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, file_name)
    info_save_path = save_path + ".info"

    data_id_man = DataIDManager()
    inst_list = generate_fn(data_id_man)
    encode_fn = get_text_pair_encode_fn(max_seq_length, tokenizer)

    write_records_w_encode_fn(save_path, encode_fn, inst_list)
    json.dump(data_id_man.id_to_info, open(info_save_path, "w"))


def generate_tfrecord_inner(save_name, tokenizer):
    data: List[Tuple[int, List[Tuple[str, str]]]] = load_alamri1_all()

    def get_generator(data_id_manager: DataIDManager):
        for group_no, pairs in data:
            for inner_idx, (t1, t2) in enumerate(pairs):
                pass
                data_id = data_id_manager.assign({
                    'group_no': group_no,
                    'inner_idx': inner_idx,
                    'text1': t1,
                    'text2': t2,
                })
                yield PairedInstance(t1, t2, data_id, 0)

    generate_and_write(save_name, get_generator, tokenizer)


def main():
    tokenizer = get_tokenizer()
    save_name = "bert_alamri1"
    generate_tfrecord_inner(save_name, tokenizer)

    # tokenizer = get_biobert_tokenizer()
    # save_name = "biobert_alamri1"
    # generate_tfrecord_inner(save_name, tokenizer)


if __name__ == "__main__":
    main()