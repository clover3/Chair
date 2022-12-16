import os

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import SingleEncoder, EvenSplitEncoder, SpacySplitEncoder, \
    SpacySplitEncoder2, SpacySplitEncoderSlash, SpacySplitEncoderNoMask, EvenSplitEncoderAvoidCut, SplitBySegmentIDs, \
    UnEvenSlice, TwoSegConcatEncoder
from data_generator2.segmented_enc.segmented_tfrecord_gen import get_encode_fn_from_encoder_list, \
    get_encode_fn_from_encoder
from dataset_specific.mnli.snli_reader_tfds import SNLIReaderTFDS
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from cpath import output_path
from misc_lib import path_join





def gen_concat_two_seg(reader, data_name, split):
    output_dir = path_join(output_path, "tfrecord", data_name)
    exist_or_mkdir(output_dir)
    save_path = os.path.join(output_dir, split)
    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoder(tokenizer, 300 * 2)
    encode_fn = get_encode_fn_from_encoder(encoder)
    write_records_w_encode_fn(save_path, encode_fn, reader.load_split(split), reader.get_data_size(split))


def main():
    reader = SNLIReaderTFDS()
    data_name = "snli_sg1"
    gen_concat_two_seg(reader, data_name, "test")
    for split in ["validation", "train"]:
        gen_concat_two_seg(reader, data_name, split)



if __name__ == "__main__":
    main()
