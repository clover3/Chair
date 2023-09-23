import pickle
from typing import List

from data_generator2.segmented_enc.es_nli.common import HSegmentedPair
from data_generator2.segmented_enc.es_nli.compute_delete_indices_by_attn import compute_attn_sel_delete_indices
from data_generator2.segmented_enc.es_nli.path_helper import get_evidence_selected0_path
from dataset_specific.mnli.mnli_reader import MNLIReader
from trainer_v2.custom_loop.attention_helper.attention_extractor_interface import AttentionExtractor
from trainer_v2.custom_loop.attention_helper.model_shortcut import load_nli14_attention_extractor
from trainer_v2.custom_loop.attention_helper.runner.attn_reasonable import enum_segmentations


def do_for_split(attn_extractor, split):
    reader = MNLIReader()
    nli_pair_iter = reader.load_split(split)
    batch_size = 16
    data_size = reader.get_data_size(split)
    itr: List[HSegmentedPair] = enum_segmentations(nli_pair_iter)
    output = list(compute_attn_sel_delete_indices(
        attn_extractor, itr, data_size, batch_size))
    save_path = get_evidence_selected0_path(split)
    pickle.dump(output, open(save_path, "wb"))


def main():
    attn_extractor: AttentionExtractor = load_nli14_attention_extractor()
    for split in ["dev", "train"]:
        do_for_split(attn_extractor, split)


if __name__ == "__main__":
    main()