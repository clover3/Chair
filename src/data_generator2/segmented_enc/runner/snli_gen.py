import os

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import TwoSegConcatEncoder, BasicConcatEncoder
from data_generator2.segmented_enc.segmented_tfrecord_gen import get_encode_fn_from_encoder_list, \
    get_encode_fn_from_encoder
from dataset_specific.mnli.snli_reader_tfds import SNLIReaderTFDS
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from cpath import output_path
from misc_lib import path_join
from trainer_v2.chair_logging import c_log


def gen_concat_two_seg(reader, encoder, data_name, split):
    output_dir = path_join(output_path, "tfrecord", data_name)
    exist_or_mkdir(output_dir)
    save_path = os.path.join(output_dir, split)
    encode_fn = get_encode_fn_from_encoder(encoder)
    write_records_w_encode_fn(save_path, encode_fn, reader.load_split(split), reader.get_data_size(split))


def do_snli_tfrecord_gen(data_name, encoder):
    c_log.info("Generating {}".format(data_name))
    reader = SNLIReaderTFDS()
    for split in ["validation", "train", "test"]:
        gen_concat_two_seg(reader, encoder, data_name, split)


def main():
    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoder(tokenizer, 300 * 2)
    data_name = "snli_sg1"
    do_snli_tfrecord_gen(data_name, encoder)


def main():
    tokenizer = get_tokenizer()
    encoder = BasicConcatEncoder(tokenizer, 300)
    data_name = "snli1"
    do_snli_tfrecord_gen(data_name, encoder)


def step_cal():
    reader = SNLIReaderTFDS()

    batch_size = 16
    data_size = reader.get_data_size("train")
    for i in range(1, 5):
        n_step = int(data_size * i / batch_size)
        print(f"{i} epoch will have {n_step} steps")


if __name__ == "__main__":
    step_cal()
