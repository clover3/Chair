import json
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.annotation_1.label_processor import json_dict_list_to_annots
from contradiction.medical_claims.label_structure import PairedIndicesLabel, AlamriLabelUnitT
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem
from cpath import output_path
from misc_lib import path_join


def load_annotation_from_json(name) -> List[AlamriLabelUnitT]:
    source_json_path = path_join(output_path, "alamri_annotation1",
                                 "label", name + ".json")
    maybe_list = json.load(open(source_json_path, "r"))
    labels: List[AlamriLabelUnitT] = json_dict_list_to_annots(maybe_list)
    return labels


def get_sent_len_info() -> Dict[Tuple[int, int], Tuple[int, int]]:
    all_problems = load_alamri_problem()
    len_info: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for p in all_problems:
        pair_id = p.group_no, p.inner_idx
        l1 = len(p.text1.split())
        l2 = len(p.text2.split())
        len_info[pair_id] = l1, l2
    return len_info


err_flag = False
def get_binary_array(indices, seq_len):
    for i in indices:
        if i >= seq_len:
            global err_flag
            err_flag = True
            print(f"Index {i} is unexpected for sequence length {seq_len}")

    return [i in indices for i in range(seq_len)]


def load_label_as_binary_array(name) -> List[Tuple[Tuple[int, int], Dict[str, List[bool]]]]:
    print(f"Loading label for {name}")
    alu_list: List[AlamriLabelUnitT] = load_annotation_from_json(name)
    len_info: Dict[Tuple[int, int], Tuple[int, int]] = get_sent_len_info()
    label_set_list: List[Tuple[Tuple[int, int], Dict[str, List[bool]]]] = []
    global err_flag

    for pair_key, labels in alu_list:
        l1, l2 = len_info[pair_key]
        err_flag = False
        bin_prem_mismatch = get_binary_array(labels.prem_mismatch_indices, l1)
        bin_hypo_mismatch = get_binary_array(labels.hypo_mismatch_indices, l2)
        bin_prem_conflict = get_binary_array(labels.prem_conflict_indices, l1)
        bin_hypo_conflict = get_binary_array(labels.hypo_conflict_indices, l2)
        d = {
            'prem_mismatch': bin_prem_mismatch,
            'hypo_mismatch': bin_hypo_mismatch,
            'prem_conflict': bin_prem_conflict,
            'hypo_conflict': bin_hypo_conflict,
        }
        label_set_list.append((pair_key, d))
        if err_flag:
            print("from ", pair_key)
    return label_set_list

