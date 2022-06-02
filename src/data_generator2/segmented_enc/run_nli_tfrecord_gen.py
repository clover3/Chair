from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import SingleEncoder, EvenSplitEncoder, SpacySplitEncoder, \
    SpacySplitEncoder2, SpacySplitEncoderSlash, SpacySplitEncoderNoMask
from data_generator2.segmented_enc.segmented_tfrecord_gen import get_encode_fn_from_encoder_list
from dataset_specific.mnli.mnli_reader import MNLIReader
from dataset_specific.mnli.parsing_jobs.partition_specs import get_mnli_spacy_split_pds
from tf_util.record_writer_wrap import write_records_w_encode_fn


def mnli_encode_common(p_encoder, h_encoder, split, output_path):
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

    mnli_encode_common(p_encoder, h_encoder, split, output_path)


def gen_spacy_tokenized(split):
    output_path = at_output_dir("tfrecord", f"nli_sg3_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SpacySplitEncoder(tokenizer, 100)
    mnli_encode_common(p_encoder, h_encoder, split, output_path)


def gen_spacy_tokenize2(split):
    pds = get_mnli_spacy_split_pds(split)
    split_d = dict(pds.read_pickles_as_itr())
    output_path = at_output_dir("tfrecord", f"nli_sg3_2_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SpacySplitEncoder2(tokenizer, 100, split_d)
    mnli_encode_common(p_encoder, h_encoder, split, output_path)


def gen_spacy_tokenize_slash(split):
    print("gen_spacy_tokenize_slash")
    pds = get_mnli_spacy_split_pds(split)
    split_d = dict(pds.read_pickles_as_itr())
    output_path = at_output_dir("tfrecord", f"nli_sg4_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SpacySplitEncoderSlash(tokenizer, 100, split_d)
    mnli_encode_common(p_encoder, h_encoder, split, output_path)



def gen_spacy_tokenize_no_mask(split):
    print("gen_spacy_tokenize_no_mask")
    pds = get_mnli_spacy_split_pds(split)
    split_d = dict(pds.read_pickles_as_itr())
    output_path = at_output_dir("tfrecord", f"nli_sg5_{split}")
    tokenizer = get_tokenizer()
    p_encoder = SingleEncoder(tokenizer, 200)
    h_encoder = SpacySplitEncoderNoMask(tokenizer, 100, split_d)
    mnli_encode_common(p_encoder, h_encoder, split, output_path)


def main():
    gen_spacy_tokenize_no_mask("dev")
    gen_spacy_tokenize_no_mask("train")


if __name__ == "__main__":
    main()
