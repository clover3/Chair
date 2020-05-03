from collections import OrderedDict
from collections import OrderedDict
from typing import Tuple, Dict, Iterator

from scipy.special import softmax

from arg.perspectives.types import CPIDPair, Logits, DataID
from data_generator.special_tokens import CLS_ID, SEP_ID
from tlm.data_gen.base import get_basic_input_feature_as_list_all_ids, ordered_dict_from_input_segment_mask_ids
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def combine_segment(features_c, features_p) -> OrderedDict:
    # input_ids does not contain CLS, SEP
    c_seg_id = list(take(features_c['segment_ids']))
    max_seq_len = len(c_seg_id)

    st = c_seg_id.index(1)
    input_mask = list(take(features_c['input_mask']))

    ed = input_mask.index(0)

    feature_c_input_ids = take(features_c['input_ids'])
    paragraph = feature_c_input_ids[st:ed]

    c_input_ids = feature_c_input_ids[:st]

    feature_p_input_ids = take(features_p['input_ids'])
    p_seg_id = list(take(features_p['segment_ids']))

    st = p_seg_id.index(1)
    p_input_ids = feature_p_input_ids[:st]

    input_ids = [CLS_ID] + c_input_ids + p_input_ids + [SEP_ID] + paragraph + [SEP_ID] #+ [random.randint(10, 13)]
    segment_ids = [0] * (2 + len(c_input_ids) + len(p_input_ids)) + [1] * (1 + len(paragraph))
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list_all_ids(input_ids, segment_ids, max_seq_len)
    return ordered_dict_from_input_segment_mask_ids(input_ids, input_mask, segment_ids)


def rel_filter(tfrecord_itr,
               relevance_scores: Dict[DataID, Tuple[CPIDPair, Logits, Logits]],
               cpid_to_label: Dict[CPIDPair, int]) -> Iterator[OrderedDict]:

    last_feature = None
    for features in tfrecord_itr:
        if last_feature is None:
            last_feature = features
            continue

        data_id = take(features["data_id"])[0]
        t = relevance_scores[data_id]
        cpid: CPIDPair = t[0]
        c_logits = t[1]
        p_logits = t[2]

        c_score = softmax(c_logits)[1]
        p_score = softmax(p_logits)[1]

        weight = c_score * p_score
        label: int = cpid_to_label[cpid]

        if weight > 0.5:
            new_feature = combine_segment(last_feature, features)
            #new_feature['weight'] = create_float_feature([weight])
            new_feature['label_ids'] = create_int_feature([label])
            new_feature['data_id'] = create_int_feature([data_id])
            yield new_feature
        last_feature = None
