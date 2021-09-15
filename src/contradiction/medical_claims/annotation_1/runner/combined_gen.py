import json
import os
from typing import List

from contradiction.medical_claims.annotation_1.label_processor import combine_alamri1_annots, \
    convert_annots_to_json_serializable
from contradiction.medical_claims.annotation_1.mturk_scheme import AlamriLabelUnit
from contradiction.medical_claims.annotation_1.runner.agreement_analysis import load_annots_w_processing
from cpath import output_path


def main():
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


if __name__ == "__main__":
    main()