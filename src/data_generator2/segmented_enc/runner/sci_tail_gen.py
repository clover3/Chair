from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.segmented_tfrecord_gen import gen_concat_two_seg
from data_generator2.segmented_enc.seg_encoder_common import TwoSegConcatEncoder, BasicConcatEncoder, NSegConcatEncoder
from dataset_specific.mnli.sci_tail import SciTailReaderTFDS


def do_sci_tail_tfrecord_gen(data_name, encoder):
    reader = SciTailReaderTFDS()
    for split in ["validation", "train", "test"]:
        gen_concat_two_seg(reader, encoder, data_name, split)


def main():
    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoder(tokenizer, 300 * 2)
    data_name = "sci_tail_sg1"
    do_sci_tail_tfrecord_gen(data_name, encoder)



def main():
    tokenizer = get_tokenizer()
    encoder = BasicConcatEncoder(tokenizer, 300)
    data_name = "sci_tail1"
    do_sci_tail_tfrecord_gen(data_name, encoder)


def sci_tail_sg2():
    tokenizer = get_tokenizer()
    encoder = NSegConcatEncoder(tokenizer, 600, 4)
    data_name = "sci_tail_sg2"
    do_sci_tail_tfrecord_gen(data_name, encoder)


if __name__ == "__main__":
    sci_tail_sg2()
