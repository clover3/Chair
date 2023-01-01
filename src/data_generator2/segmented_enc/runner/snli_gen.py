from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.segmented_tfrecord_gen import gen_concat_two_seg
from data_generator2.segmented_enc.seg_encoder_common import TwoSegConcatEncoder, BasicConcatEncoder, NSegConcatEncoder
from dataset_specific.mnli.snli_reader_tfds import SNLIReaderTFDS
from trainer_v2.chair_logging import c_log


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


def snli1():
    tokenizer = get_tokenizer()
    encoder = BasicConcatEncoder(tokenizer, 300)
    data_name = "snli1"
    do_snli_tfrecord_gen(data_name, encoder)


def snli_sg2():
    tokenizer = get_tokenizer()
    encoder = NSegConcatEncoder(tokenizer, 600, 4)
    data_name = "snli_sg2"
    do_snli_tfrecord_gen(data_name, encoder)


def step_cal():
    reader = SNLIReaderTFDS()

    batch_size = 16
    data_size = reader.get_data_size("train")
    for i in range(1, 5):
        n_step = int(data_size * i / batch_size)
        print(f"{i} epoch will have {n_step} steps")


if __name__ == "__main__":
    snli_sg2()
