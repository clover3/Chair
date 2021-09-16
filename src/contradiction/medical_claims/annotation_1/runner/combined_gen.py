import json
import os
from typing import List

from contradiction.medical_claims.annotation_1.label_processor import combine_alamri1_annots, \
    convert_annots_to_json_serializable
from contradiction.medical_claims.annotation_1.mturk_scheme import AlamriLabelUnit
from contradiction.medical_claims.annotation_1.process_annotation import load_annots_w_processing, \
    load_annots_for_worker, trusted_worker
from cpath import output_path


def sel_by_longest():
    annots: List[AlamriLabelUnit] = load_annots_w_processing()

    def not6(e):
        (group_no, idx), annot = e
        return group_no != 6

    annots = list(filter(not6, annots))

    def num_tokens(e: AlamriLabelUnit):
        (group_no, idx), annot = e
        return sum(map(len, annot.enum()))

    def combine_method(entries):
        entries.sort(key=num_tokens, reverse=True)
        final_e = entries[0]
        return final_e

    out = combine_alamri1_annots(annots, combine_method)
    out = convert_annots_to_json_serializable(out)
    save_dir = os.path.join(output_path, "alamri_annotation1", "label", "sel_by_longest.json")
    json.dump(out, open(save_dir, "w"), indent=True)


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
    sel_by_worker_id()