import os

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import SingleEncoder, EvenSplitEncoder, SpacySplitEncoder, \
    SpacySplitEncoder2, SpacySplitEncoderSlash, SpacySplitEncoderNoMask, EvenSplitEncoderAvoidCut, SplitBySegmentIDs, \
    UnEvenSlice
from data_generator2.segmented_enc.segmented_tfrecord_gen import get_encode_fn_from_encoder_list
from dataset_specific.mnli.mnli_reader import MNLIReader
from dataset_specific.mnli.parsing_jobs.partition_specs import get_mnli_spacy_split_pds
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path):
    data_size = 400 * 1000 if split == "train" else 10000
    reader = MNLIReader()
    encode_fn = get_encode_fn_from_encoder_list([p_encoder, h_encoder])
    write_records_w_encode_fn(output_path, encode_fn, reader.load_split(split), data_size)
    print("p has {} warning cases".format(p_encoder.counter_warning.cnt))
    print("h has {} warning cases".format(h_encoder.counter_warning.cnt))


def gen_mnli(split):
    output_path = at_output_dir("tfrecord", f"nli_p200_hseg2_100_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = EvenSplitEncoder(tokenizer, 100)

    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)



def gen_spacy_tokenized(split):
    output_path = at_output_dir("tfrecord", f"nli_sg3_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SpacySplitEncoder(tokenizer, 100)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)


def gen_spacy_tokenize2(split):
    pds = get_mnli_spacy_split_pds(split)
    split_d = dict(pds.read_pickles_as_itr())
    output_path = at_output_dir("tfrecord", f"nli_sg3_2_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SpacySplitEncoder2(tokenizer, 100, split_d)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)


def gen_spacy_tokenize_slash(split):
    print("gen_spacy_tokenize_slash")
    pds = get_mnli_spacy_split_pds(split)
    split_d = dict(pds.read_pickles_as_itr())
    output_path = at_output_dir("tfrecord", f"nli_sg4_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SpacySplitEncoderSlash(tokenizer, 100, split_d)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)


def gen_spacy_tokenize_no_mask(split):
    print("gen_spacy_tokenize_no_mask")
    pds = get_mnli_spacy_split_pds(split)
    split_d = dict(pds.read_pickles_as_itr())
    output_path = at_output_dir("tfrecord", f"nli_sg5_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SpacySplitEncoderNoMask(tokenizer, 100, split_d)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)


def gen_mnli_avoid_cut(split):
    output_dir = at_output_dir("tfrecord", "nli_sg6")
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = EvenSplitEncoderAvoidCut(tokenizer, 100)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)


def gen_mnli_avoid_cut_split_by_seg_id(split):
    output_dir = at_output_dir("tfrecord", "nli_sg7")
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SplitBySegmentIDs(tokenizer, 100)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)


def gen_mnli_un_even(split):
    output_dir = at_output_dir("tfrecord", "nli_sg8")
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = UnEvenSlice(tokenizer, 200)
    mnli_asymmetric_encode_common(p_encoder, h_encoder, split, output_path)


def main():
    gen_mnli_un_even("dev")
    gen_mnli_un_even("train")


if __name__ == "__main__":
    main()
