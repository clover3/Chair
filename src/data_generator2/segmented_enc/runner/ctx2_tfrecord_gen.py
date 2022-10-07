import os

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.runner.run_nli_tfrecord_gen import mnli_asymmetric_encode_common
from data_generator2.segmented_enc.seg_encoder_common import SingleChunkIndicatingEncoder, ChunkIndicatingEncoder
from misc_lib import exist_or_mkdir
from tf_util.lib.tf_funcs import show_tfrecord


def gen_single_chunk_indicating_encoder(split):
    output_dir = at_output_dir("tfrecord", "nli_sg10")
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    tokenizer = get_tokenizer()
    p_encoder = SingleChunkIndicatingEncoder(tokenizer, 200)
    h_encoder = SingleChunkIndicatingEncoder(tokenizer, 100)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)
    print("{} of tokens are chunk start".format(h_encoder.chunk_start_rate.get_suc_prob()))



def gen_chunk_indicating_encoder(split):
    output_dir = at_output_dir("tfrecord", "nli_sg11")
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    tokenizer = get_tokenizer()
    typical_chunk_len = 4
    p_encoder = ChunkIndicatingEncoder(tokenizer, 200, typical_chunk_len)
    h_encoder = ChunkIndicatingEncoder(tokenizer, 100, typical_chunk_len)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)
    print("{} of tokens are chunk start".format(h_encoder.chunk_start_rate.get_suc_prob()))
    show_tfrecord(output_path)


def do_train_dev(fn):
    for split in ["dev", "train"]:
        fn(split)


def main():
    do_train_dev(gen_chunk_indicating_encoder)


if __name__ == "__main__":
    main()