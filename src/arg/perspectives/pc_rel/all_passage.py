from collections import OrderedDict
from typing import Tuple, Dict, Iterator, List, NamedTuple

from scipy.special import softmax

from arg.perspectives.types import CPIDPair, Logits, DataID
from list_lib import flatten
from misc_lib import group_by
from tlm.data_gen.base import ordered_dict_from_input_segment_mask_ids
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def extract_paragraph(features_c) -> List[int]:
    # input_ids does not contain CLS, SEP
    segment_ids = list(take(features_c['segment_ids']))
    input_mask = list(take(features_c['input_mask']))
    ed, st = get_second_seg_location(segment_ids, input_mask)
    input_ids = take(features_c['input_ids'])
    paragraph = input_ids[st:ed]
    return paragraph


def extract_paragraph_score(entry) -> Tuple[List[int], List[float]]:
    # input_ids does not contain CLS, SEP
    input_ids, input_mask, segment_ids, data_id, scores = entry
    ed, st = get_second_seg_location(segment_ids, input_mask)

    return input_ids[st:ed], scores[st:ed]


def get_second_seg_location(segment_ids, input_mask):
    st = segment_ids.index(1)
    ed = input_mask.index(0)
    return ed, st


class PCEvidence(NamedTuple):
    paragraph_tokens : List[int]
    cpid : CPIDPair
    label: int
    data_id: int
    focus_mask: List[int]


def remove_duplicate(evidence_list: List[PCEvidence]):
    def para_hash(evidence: PCEvidence):
        return " ".join([str(term) for term in evidence.paragraph_tokens])

    hash_set = set()

    for evidence in evidence_list:
        hash = para_hash(evidence)
        if hash in hash_set:
            continue

        hash_set.add(hash)
        yield evidence


def combine_paragraph(pargraph_list: List[PCEvidence],
                      num_max_para,
                      max_seq_length):
    # 1. Remove duplicate
    # 2. concatenate paragraphs
    # 3. Pad to max length
    result_list = []
    input_mask_list = []
    for evidence in remove_duplicate(pargraph_list):
        new_para = evidence.paragraph_tokens[:max_seq_length]
        input_mask = len(new_para) * [1]
        num_pad = max_seq_length - len(new_para)
        new_para += num_pad * [0]
        input_mask += num_pad * [0]
        result_list.append(new_para)
        input_mask_list.append(input_mask)
        if len(result_list) == num_max_para:
            break

    return list(flatten(result_list)), list(flatten(input_mask_list))


def extract_rel_score_and_combine(paragraph_tokens: List, c_score_info: List, p_score_info: List):
    c_input_ids_slice, c_scores = extract_paragraph_score(c_score_info)
    p_input_ids_slice, p_scores = extract_paragraph_score(p_score_info)

    k = len(p_input_ids_slice)

    assert len(p_input_ids_slice) == len(paragraph_tokens)
    assert len(c_input_ids_slice) == len(paragraph_tokens)


    combined_scores = []
    for cs, ps in zip(c_scores, p_scores):
        combined_scores.append(max(cs, ps))
    assert len(combined_scores) == k

    sorted_combined_scores = combined_scores.copy()
    sorted_combined_scores.sort(reverse=True)
    assert len(sorted_combined_scores) == k
    assert len(combined_scores) == k
    cut_portion = 0.2
    cut_score = sorted_combined_scores[int(len(sorted_combined_scores ) * cut_portion)]

    focus_mask = list([1 if s > cut_score else 0 for s in combined_scores])
    assert len(focus_mask) == k
    return focus_mask


def collect_passages(tfrecord_itr,
                     relevance_scores: Dict[DataID, Tuple[CPIDPair, Logits, Logits]],
                     cpid_to_label: Dict[CPIDPair, int],
                     num_max_para: int,
                     window_size: int,
                     token_rel_score_info: Dict[DataID, List] = None
                     ) -> Iterator[OrderedDict]:

    cpid_paragraph = []

    c_arr = []
    p_arr = []
    for idx, features in enumerate(tfrecord_itr):
        if idx % 2 == 0:  # CID entry
            c_arr.append(features)
        else:
            p_arr.append(features)

    print("number of (cpid, paragraph)", len(c_arr))
    assert len(c_arr) == len(p_arr)
    for i in range(len(c_arr)):
        c_features = c_arr[i]
        p_features = p_arr[i]

        c_data_id = take(c_features["data_id"])[0]
        p_data_id = take(p_features["data_id"])[0]

        t = relevance_scores[c_data_id]
        cpid: CPIDPair = t[0]
        c_logits = t[1]
        p_logits = t[2]
        c_score = softmax(c_logits)[1]
        p_score = softmax(p_logits)[1]

        weight = c_score * p_score
        label: int = cpid_to_label[cpid]

        if weight > 0.5:
            paragraph = extract_paragraph(c_features)

            if token_rel_score_info is not None:
                focus_mask = extract_rel_score_and_combine(
                    paragraph,
                    token_rel_score_info[c_data_id],
                    token_rel_score_info[p_data_id])
            else:
                focus_mask = [0] * len(paragraph)

            e = PCEvidence(paragraph_tokens=paragraph,
                           cpid=cpid,
                           label=label,
                           data_id=c_data_id,
                           focus_mask=focus_mask
                           )

            cpid_paragraph.append(e)
    add_focus_mask = token_rel_score_info is not None

    for cpid, evidence_list in group_by(cpid_paragraph, lambda pc_evidence: pc_evidence.cpid).items():
        if evidence_list:
            first_item = evidence_list[0]
            new_feature = build_new_feature(evidence_list,
                                            first_item.label,
                                            first_item.data_id,
                                            num_max_para,
                                            window_size,
                                            add_focus_mask)
            yield new_feature


def build_new_feature(evidence_list: List[PCEvidence], label, data_id, num_max_para, window_size, add_focus_mask=False):
    input_ids_list: List[List] = []
    input_mask_list = []
    focus_mask_list = []
    for evidence in remove_duplicate(evidence_list):
        new_para = evidence.paragraph_tokens[:window_size]
        input_mask = len(new_para) * [1]
        focus_mask = evidence.focus_mask[:window_size]

        assert len(focus_mask) == len(new_para)

        num_pad = window_size - len(new_para)
        new_para += num_pad * [0]
        input_mask += num_pad * [0]
        focus_mask += num_pad * [0]

        input_ids_list.append(new_para)
        input_mask_list.append(input_mask)
        focus_mask_list.append(focus_mask)
        if len(input_ids_list) == num_max_para:
            break

    total_len = num_max_para * window_size

    def flatten_and_drop_tail(id_list_list: List[List[int]]):
        out_ids = list(flatten(id_list_list))
        out_ids += [0] * (total_len - len(out_ids))
        return out_ids

    input_ids = flatten_and_drop_tail(input_ids_list)
    input_mask = flatten_and_drop_tail(input_mask_list)
    segment_ids = len(input_ids) * [0]

    new_feature = ordered_dict_from_input_segment_mask_ids(input_ids, input_mask, segment_ids)

    if add_focus_mask:
        focus_mask = flatten_and_drop_tail(focus_mask_list)
        new_feature['focus_mask'] = create_int_feature(focus_mask)

    new_feature['label_ids'] = create_int_feature([label])
    new_feature['data_id'] = create_int_feature([data_id])
    return new_feature

