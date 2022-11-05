from typing import List, Iterable

from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from dataset_specific.mnli.mnli_reader import MNLIReader, NLIPairData
from trainer_v2.custom_loop.attention_helper.attention_extractor import AttentionExtractor, AttentionScoresDetailed
from trainer_v2.custom_loop.attention_helper.evidence_selector_0 import get_delete_indices
from data_generator2.segmented_enc.es.common import HSegmentedPair
from trainer_v2.custom_loop.attention_helper.model_shortcut import load_nli14_attention_extractor


def enum_segmentations(nli_pair_iter: Iterable[NLIPairData]) -> List[HSegmentedPair]:
    tokenizer = get_tokenizer()
    for item in nli_pair_iter:
        p_tokens = tokenizer.tokenize(item.premise)
        h_tokens = tokenizer.tokenize(item.hypothesis)
        st, ed = get_random_split_location(h_tokens)
        yield HSegmentedPair(p_tokens, h_tokens, st, ed, item)


def main():
    attn_extractor: AttentionExtractor = load_nli14_attention_extractor()
    # For random segmentations of h tokens
    split = "train"
    reader = MNLIReader()
    nli_pair_iter = reader.load_split(split)
    for e in enum_segmentations(nli_pair_iter):
        res = attn_extractor.predict_list([(e.p_tokens, e.h_tokens)])
        attn_score: AttentionScoresDetailed = res[0]
        attn_merged = attn_score.get_layer_head_merged()
        delete_indices_list = get_delete_indices(attn_merged, e)
        h_tokens_i = [e.get_first_h_tokens_w_mask(), e.get_second_h_tokens()]

        print(f"Premise: {e.nli_pair.premise}")
        for i in [0, 1]:
            p_tokens_copy = list(e.p_tokens)
            for j in delete_indices_list[i]:
                p_tokens_copy[j] = "[MASK]"

            print("P{}: {}".format(i+1, pretty_tokens(p_tokens_copy, True)))
            print("H{}: {}".format(i+1, pretty_tokens(h_tokens_i[i], True)))

        _ = input("Enter something to continue")



if __name__ == "__main__":
    main()