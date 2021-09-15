import json
import os
from typing import List

from contradiction.medical_claims.annotation_1.label_processor import json_dict_list_to_annots, save_annots_to_qrel
from contradiction.medical_claims.annotation_1.mturk_scheme import AlamriLabelUnit
from cpath import output_path


def convert_save_to_trec_format(name):
    source_json_path = os.path.join(output_path, "alamri_annotation1", "label", name + ".json")
    save_path = os.path.join(output_path, "alamri_annotation1", "label", name + ".qrel")
    maybe_list = json.load(open(source_json_path, "r"))
    labels: List[AlamriLabelUnit] = json_dict_list_to_annots(maybe_list)
    save_annots_to_qrel(labels, save_path)


def main():
    name = "sel_by_longest"
    convert_save_to_trec_format(name)


if __name__ == "__main__":
    main()
