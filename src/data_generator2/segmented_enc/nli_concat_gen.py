import os

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import TwoSegConcatEncoder
from data_generator2.segmented_enc.segmented_tfrecord_gen import get_encode_fn_from_encoder
from dataset_specific.mnli.mnli_reader import MNLIReader
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def mnli_encode_common(encode_fn, split, output_path):
    data_size = 400 * 1000 if split == "train" else 10000
    reader = MNLIReader()
    write_records_w_encode_fn(output_path, encode_fn, reader.load_split(split), data_size)


def gen_mnli_concat_two_seg(split):
    output_dir = at_output_dir("tfrecord", "nli_sg8")
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoder(tokenizer, 300 * 2)
    encode_fn = get_encode_fn_from_encoder(encoder)
    mnli_encode_common(encode_fn, split, output_path)


def main():
    for split in ["dev", "train"]:
        gen_mnli_concat_two_seg(split)


if __name__ == "__main__":
    main()
