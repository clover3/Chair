from typing import List, Tuple, NamedTuple

from alignment import RelatedEvalInstance
from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from bert_api.task_clients.nli_interface.nli_interface import NLIInput


def pairwise_feature(text1: SegmentedText,
                     text2: SegmentedText,
                     seg1_target_idx,
                     seg2_target_idx,
                     ) -> List[SegmentedInstance]:
    # text1 is query / hypothesis
    q_wo_qt: SegmentedText = text1.get_dropped_text([seg1_target_idx])
    qt: SegmentedText = text1.get_sliced_text([seg1_target_idx])

    d_wo_dt: SegmentedText = text2.get_dropped_text([seg2_target_idx])
    dt: SegmentedText = text2.get_sliced_text([seg2_target_idx])

    feature_runs: List[SegmentedInstance] = []
    for new_text1 in [text1, q_wo_qt, qt]:
        for new_text2 in [text2, d_wo_dt, dt]:
            feature_runs.append(SegmentedInstance(new_text1, new_text2))

    assert len(feature_runs) == 9
    return feature_runs


def pairwise_feature_ex(text1: SegmentedText,
                        text2: SegmentedText,
                        seg1_target_indices,
                        seg2_target_indices,
                        ) -> List[SegmentedInstance]:
    # text1 is query / hypothesis
    q_wo_qt: SegmentedText = text1.get_dropped_text(seg1_target_indices)
    qt: SegmentedText = text1.get_sliced_text(seg1_target_indices)

    d_wo_dt: SegmentedText = text2.get_dropped_text(seg2_target_indices)
    dt: SegmentedText = text2.get_sliced_text(seg2_target_indices)

    feature_runs: List[SegmentedInstance] = []
    for new_text1 in [text1, q_wo_qt, qt]:
        for new_text2 in [text2, d_wo_dt, dt]:
            feature_runs.append(SegmentedInstance(new_text1, new_text2))

    assert len(feature_runs) == 9
    return feature_runs


def pairwise_feature4(text1: SegmentedText,
                     text2: SegmentedText,
                     seg1_target_idx,
                     seg2_target_idx,
                     ) -> List[SegmentedInstance]:
    # text1 is query / hypothesis
    q_wo_qt: SegmentedText = text1.get_dropped_text([seg1_target_idx])
    d_wo_dt: SegmentedText = text2.get_dropped_text([seg2_target_idx])

    feature_runs: List[SegmentedInstance] = []
    for new_text1 in [text1, q_wo_qt]:
        for new_text2 in [text2, d_wo_dt]:
            feature_runs.append(SegmentedInstance(new_text1, new_text2))

    assert len(feature_runs) == 4
    return feature_runs


Features = List[SegmentedInstance]
class RankList(NamedTuple):
    seg_instance: SegmentedInstance
    sub_id: str
    rank_item_list: List[Tuple[int, Features]]


def perturbation_enum(si: SegmentedInstance) -> List[RankList]:
    text1 = si.text1
    text2 = si.text2
    problem_list_per_pair = []
    for seg1_idx in text1.enum_seg_idx():
        # Goal Rank Candidates
        rank_item_list: List[Tuple[int, Features]] = []
        for seg2_idx in text2.enum_seg_idx():
            item_id = seg2_idx
            features = pairwise_feature(text1, text2, seg1_idx, seg2_idx)
            rank_item = item_id, features
            rank_item_list.append(rank_item)

        problem_list_per_pair.append(
            RankList(si, str(seg1_idx), rank_item_list))
    return problem_list_per_pair


def eval_all_perturbations(nli_client, problems: List[RelatedEvalInstance]):
    hash_key_set = set()
    for idx, p in enumerate(problems):
        # tprint("problem {}".format(idx))
        ri_list: List[RankList] = perturbation_enum(p.seg_instance)

        for item in ri_list:
            for _, si_list in item.rank_item_list:
                todo = []
                for si in si_list:
                    nli_input = NLIInput(si.text2, si.text1)
                    hash_key = nli_input.str_hash()
                    if hash_key not in hash_key_set:
                        hash_key_set.add(hash_key)
                        todo.append(nli_input)

                nli_client.predict(todo)