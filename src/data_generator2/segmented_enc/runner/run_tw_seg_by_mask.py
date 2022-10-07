import os

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.gs_uploader import upload_nli_sg_files
from data_generator2.segmented_enc.mnli_common import mnli_encode_common
from data_generator2.segmented_enc.seg_encoder_common import TwoSegConcatEncoder
from data_generator2.segmented_enc.segmented_tfrecord_gen import get_encode_fn_from_encoder
from data_generator2.segmented_enc.two_seg_by_mask import ConcatWMask
from misc_lib import exist_or_mkdir


def encode_w_mask(data_name, split):
    output_dir = at_output_dir("tfrecord", data_name)
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    tokenizer = get_tokenizer()
    encoder = ConcatWMask(tokenizer, 300)
    encode_fn = get_encode_fn_from_encoder(encoder)
    mnli_encode_common(encode_fn, split, output_path)


def main():
    data_name = "nli_sg12"
    for split in ["dev", "train"]:
        encode_w_mask(data_name, split)


if __name__ == "__main__":
    main()
