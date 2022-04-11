from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple


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


def main():
    # TODO
    #   Given a pair instance.
    #   Run all perturbations

    return NotImplemented


if __name__ == "__main__":
    main()