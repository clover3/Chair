from typing import List, Tuple, Dict

from contradiction.medical_claims.annotation_1.mturk_scheme import PairedIndicesLabel
from misc_lib import group_by, get_first


def combine_alamri1_annots(annots, combine_method) -> List[Tuple[Tuple[int, int], PairedIndicesLabel]]:
    grouped = group_by(annots, get_first)


    selected_annots = []
    for hit_id, entries in grouped.items():
        final_e = combine_method(entries)
        selected_annots.append(final_e)

    def to_dict(e: Tuple[Tuple[int, int], PairedIndicesLabel]) -> Dict:
        (group_no, idx), annot = e
        return {
            'group_no': group_no,
            'inner_idx': idx,
            'label': annot.to_dict()
        }

    out = list(map(to_dict, selected_annots))
    return out