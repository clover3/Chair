import os

from transformers import AutoTokenizer

from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.splade_regression.data_generator.encode_fns import get_three_text_encode_fn
from trainer_v2.per_project.transparency.splade_regression.data_loaders.pairwise_eval import load_pairwise_eval_data


def main():
    checkpoint_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)
    st = 1000
    ed = 1010
    target_partition = list(range(1000, 1010))
    num_partitions = len(target_partition)
    triplet_list = load_pairwise_eval_data(target_partition)

    def tokenize_triplet(triplet):
        t1, t2, t3 = triplet
        return tokenizer(t1), tokenizer(t2), tokenizer(t3)

    itr = map(tokenize_triplet, triplet_list)
    save_dir = path_join("output", "splade", "regression_pairwise")
    max_seq_length = 256
    save_path = os.path.join(save_dir, f"{st}_{ed}")
    encode_fn = get_three_text_encode_fn(max_seq_length)
    n_item = 10000 * num_partitions
    write_records_w_encode_fn(save_path, encode_fn, itr, n_item)




if __name__ == "__main__":
    main()
