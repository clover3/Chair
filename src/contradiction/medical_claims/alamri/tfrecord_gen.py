import json
import os
from typing import Iterator

from contradiction.medical_claims.alamri.pairwise_gen import enum_true_instance, enum_neg_instance, enum_neg_instance2, \
    enum_neg_instance_diff_review
from contradiction.medical_claims.biobert.voca_common import get_biobert_tokenizer
from cpath import at_output_dir, output_path
from data_generator.cls_sep_encoder import get_text_pair_encode_fn, PairedInstance
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import DataIDManager, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn

Entailment = 0
Neutral = 1
Contradiction = 2


def generate_true_pairs(data_id_man):
    yield from generate_inner(data_id_man, enum_true_instance)


def generate_neg_pairs(data_id_man):
    enum_fn = enum_neg_instance
    yield from generate_inner(data_id_man, enum_fn)


def generate_neg_pairs2(data_id_man) -> Iterator[PairedInstance]:
    enum_fn = enum_neg_instance2
    yield from generate_inner(data_id_man, enum_fn)


def generate_neg_pairs_diff_review(data_id_man):
    enum_fn = enum_neg_instance_diff_review
    yield from generate_inner(data_id_man, enum_fn)


def generate_inner(data_id_man, enum_fn) -> PairedInstance:
    for c1, c2, pair_type in enum_fn():
        info = {
            'text1': c1.text,
            'text2': c2.text,
            'pair_type': pair_type
        }
        inst = PairedInstance(c1.text, c2.text, data_id_man.assign(info), Neutral)
        yield inst


def generate_and_write(file_name, generate_fn, tokenizer):
    data_id_man = DataIDManager()
    inst_list = generate_fn(data_id_man)
    max_seq_length = 300
    save_path = at_output_dir("alamri_tfrecord", file_name)
    encode_fn = get_text_pair_encode_fn(max_seq_length, tokenizer)
    write_records_w_encode_fn(save_path, encode_fn, inst_list)
    info_save_path = at_output_dir("alamri_tfrecord", file_name + ".info")
    json.dump(data_id_man.id_to_info, open(info_save_path, "w"))


def bert_true_pairs():
    tokenizer = get_tokenizer()
    file_name = "bert_true_pairs"
    generate_fn = generate_true_pairs
    generate_and_write(file_name, generate_fn, tokenizer)


def bert_neg_pairs():
    tokenizer = get_tokenizer()
    file_name = "bert_neg_pairs"
    generate_fn = generate_neg_pairs
    generate_and_write(file_name, generate_fn, tokenizer)


def biobert_true_pairs():
    tokenizer = get_biobert_tokenizer()
    file_name = "biobert_true_pairs"
    generate_fn = generate_true_pairs
    generate_and_write(file_name, generate_fn, tokenizer)


def biobert_neg_pairs():
    tokenizer = get_biobert_tokenizer()
    file_name = "biobert_neg_pairs"
    generate_fn = generate_neg_pairs
    generate_and_write(file_name, generate_fn, tokenizer)


def bert_neg_pairs2():
    tokenizer = get_tokenizer()
    file_name = "bert_neg_pairs2"
    generate_fn = generate_neg_pairs2
    generate_and_write(file_name, generate_fn, tokenizer)


def biobert_neg_pairs2():
    tokenizer = get_biobert_tokenizer()
    file_name = "biobert_neg_pairs2"
    generate_fn = generate_neg_pairs2
    generate_and_write(file_name, generate_fn, tokenizer)


def bert_neg_pairs_diff_review():
    tokenizer = get_tokenizer()
    file_name = "bert_neg_pairs_diff_review"
    generate_fn = generate_neg_pairs_diff_review
    generate_and_write(file_name, generate_fn, tokenizer)


def biobert_neg_pairs_diff_review():
    tokenizer = get_biobert_tokenizer()
    file_name = "biobert_neg_pairs_diff_review"
    generate_fn = generate_neg_pairs_diff_review
    generate_and_write(file_name, generate_fn, tokenizer)


def main():
    exist_or_mkdir(os.path.join(output_path, "alamri_tfrecord"))
    bert_neg_pairs_diff_review()
    biobert_neg_pairs_diff_review()
    # bert_neg_pairs2()
    # biobert_neg_pairs2()

    # bert_true_pairs()
    # bert_neg_pairs()
    # biobert_true_pairs()
    # biobert_neg_pairs()


if __name__ == "__main__":
    main()
