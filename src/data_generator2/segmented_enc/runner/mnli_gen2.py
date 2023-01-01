from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import BasicConcatEncoder, NSegConcatEncoder
from data_generator2.segmented_enc.segmented_tfrecord_gen import gen_concat_two_seg
from dataset_specific.mnli.mnli_reader import MNLIReader


def gen_mnli():
    data_name = "mnli_150"
    tokenizer = get_tokenizer()
    encoder = BasicConcatEncoder(tokenizer, 150)
    reader = MNLIReader()
    gen_concat_two_seg(reader, encoder, data_name, "dev")
    gen_concat_two_seg(reader, encoder, data_name, "train")


def mnli_sg13():
    data_name = "mnli_sg13"
    tokenizer = get_tokenizer()
    encoder = NSegConcatEncoder(tokenizer, 600, 4)
    reader = MNLIReader()
    gen_concat_two_seg(reader, encoder, data_name, "dev")
    gen_concat_two_seg(reader, encoder, data_name, "train")


if __name__ == "__main__":
    mnli_sg13()
