
import os

from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sample_dev_as_pairs, \
    load_msmarco_sample_a_as_pairs
from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.transformers_utils import get_transformer_pair_tokenizer, get_multi_text_encode_fn

# trnsfrmr

def main():
    tuple_itr = load_msmarco_sample_a_as_pairs()
    tokenize_pair = get_transformer_pair_tokenizer()

    itr = map(tokenize_pair, tuple_itr)
    save_dir = path_join("output", "msmarco", "passage")
    max_seq_length = 256
    save_path = os.path.join(save_dir, "sample_dev1000.tfrecord")
    encode_fn = get_multi_text_encode_fn(max_seq_length, n_text=2)
    n_item = 1000 * 1000
    write_records_w_encode_fn(save_path, encode_fn, itr, n_item)



if __name__ == "__main__":
    main()

