import json
import os
from typing import List

from contradiction.medical_claims.annotation_1.label_processor import combine_alamri1_annots, \
    convert_annots_to_json_serializable
from contradiction.medical_claims.annotation_1.process_annotation import load_annots_w_processing, \
    load_annots_for_worker, load_annots_trusted
from contradiction.medical_claims.annotation_1.worker_id_info import trusted_worker
from contradiction.medical_claims.label_structure import AlamriLabelUnitT
from contradiction.medical_claims.token_tagging.path_helper import get_sbl_label_json_path, get_sbl2_label_json_path
from cpath import output_path


def sel_by_longest():
    save_dir = get_sbl_label_json_path()
    annots: List[AlamriLabelUnitT] = load_annots_w_processing()
    for pair_id, _ in annots:
        group_no, inner_idx = pair_id
        if group_no == 1:
            print(pair_id)
    sel_by_longest_inner(annots, save_dir)


def sel_by_longest_inner(annots, save_dir):
    def not6(e):
        (group_no, idx), annot = e
        return group_no != 6

    annots = list(filter(not6, annots))

    def num_tokens(e: AlamriLabelUnitT):
        (group_no, idx), annot = e
        return sum(map(len, annot.enum()))

    def combine_method(entries):
        entries.sort(key=num_tokens, reverse=True)
        final_e = entries[0]
        return final_e

    out = combine_alamri1_annots(annots, combine_method)
    out = convert_annots_to_json_serializable(out)
    json.dump(out, open(save_dir, "w"), indent=True)


def sel_by_longest2():
    save_dir = get_sbl2_label_json_path()
    annots: List[AlamriLabelUnitT] = load_annots_trusted()
    sel_by_longest_inner(annots, save_dir)


def sel_by_worker_id():
    def combine_as_verify(entries):
        assert len(entries) == 1
        final_e = entries[0]
        return final_e
    def not6(e):
        (group_no, idx), annot = e
        return group_no != 6

    for worker_id in trusted_worker:
        annots = load_annots_for_worker(worker_id)
        worker_rep = worker_id[2]
        annots = list(filter(not6, annots))

        out = combine_alamri1_annots(annots, combine_as_verify)
        out = convert_annots_to_json_serializable(out)

        save_dir = os.path.join(output_path, "alamri_annotation1", "label", "worker_{}.json".format(worker_rep))
        json.dump(out, open(save_dir, "w"), indent=True)


if __name__ == "__main__":
    # sel_by_longest()
    sel_by_worker_id()