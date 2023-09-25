from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair
from tlm.data_gen.base import combine_with_sep_cls, get_basic_input_feature_as_list, concat_triplet_windows


def concat_ph_to_encode_fn(tokenizer, segment_len, e: PHSegmentedPair):
    triplet_list = []
    for i in [0, 1]:
        tokens, segment_ids = combine_with_sep_cls(
            segment_len, e.get_partial_hypo(i), e.get_partial_prem(i))
        triplet = get_basic_input_feature_as_list(tokenizer, segment_len,
                                                  tokens, segment_ids)
        triplet_list.append(triplet)
    triplet = concat_triplet_windows(triplet_list, segment_len)
    return triplet
