import pickle
import sys
from typing import List, Iterable

from transformers import AutoTokenizer

from data_generator2.segmented_enc.es_nli.common import HSegmentedPair, PHSegmentedPair
from data_generator2.segmented_enc.es_nli.compute_delete_indices_by_attn import compute_attn_sel_delete_indices
from data_generator2.segmented_enc.es_nli.path_helper import get_evidence_selected0_path, get_mmp_es0_path
from data_generator2.segmented_enc.es_mmp.iterate_mmp import iter_train_data_as_nli_pair, \
    iter_train_triples_as_nli_pair, iter_qd_sample_as_nli_pair
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from dataset_specific.mnli.mnli_reader import NLIPairData
from trainer_v2.custom_loop.attention_helper.attention_extractor_hf import load_mmp1_attention_extractor
from trainer_v2.custom_loop.attention_helper.attention_extractor_interface import AttentionExtractor
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def enum_segmentations(nli_pair_iter: Iterable[NLIPairData]) -> Iterable[HSegmentedPair]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for item in nli_pair_iter:
        p_tokens = tokenizer.tokenize(item.premise)
        h_tokens = tokenizer.tokenize(item.hypothesis)
        st, ed = get_random_split_location(h_tokens)
        yield HSegmentedPair(p_tokens, h_tokens, st, ed, item)


def enum_segmentations2(tokenizer, item: NLIPairData):
    p_tokens = tokenizer.tokenize(item.premise)
    h_tokens = tokenizer.tokenize(item.hypothesis)
    st, ed = get_random_split_location(h_tokens)
    return HSegmentedPair(p_tokens, h_tokens, st, ed, item)


def select_delete_indices_by_attentions(
        attn_extractor, nli_pair_iter, data_size) -> Iterable[PHSegmentedPair]:
    tf_call_batch_size = 1024

    itr: Iterable[HSegmentedPair] = enum_segmentations(nli_pair_iter)
    return compute_attn_sel_delete_indices(
        attn_extractor, itr, data_size, tf_call_batch_size)


def do_for_partition(attn_extractor, partition_no):
    data_size = 30000
    nli_pair_iter = iter_qd_sample_as_nli_pair(partition_no)
    output_itr = select_delete_indices_by_attentions(
        attn_extractor, nli_pair_iter, data_size)
    output = list(output_itr)
    save_path = get_mmp_es0_path(partition_no)
    pickle.dump(output, open(save_path, "wb"))


def main(job_no):
    attn_extractor: AttentionExtractor = load_mmp1_attention_extractor()
    partition_per_job = 10
    st = job_no * partition_per_job
    ed = st + partition_per_job
    for partition_no in range(st, ed):
        try:
            print(f"Partition {partition_no}")
            do_for_partition(attn_extractor, partition_no)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    job_no = int(sys.argv[1])
    strategy = get_strategy(False, "")
    with strategy.scope():
        main(job_no)
