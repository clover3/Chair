from typing import List, Iterator

from contradiction.medical_claims.annotation_1.label_processor import load_label_from_json_path
from contradiction.medical_claims.annotation_1.load_data import get_dev_group_no, get_test_group_no
from contradiction.medical_claims.label_structure import AlamriLabelUnitT, AlamriLabel
from contradiction.medical_claims.token_tagging.path_helper import get_sbl_label_json_path


def load_sbl_labels(split="all") -> List[AlamriLabel]:
    label_list_tuple: List[AlamriLabelUnitT] = load_label_from_json_path(get_sbl_label_json_path())
    label_list: Iterator[AlamriLabel] = map(AlamriLabel.from_tuple, label_list_tuple)
    dev_group_no: List[int] = get_dev_group_no()
    test_group_no: List[int] = get_test_group_no()

    def is_dev(label: AlamriLabel) -> bool:
        return label.group_no in dev_group_no

    def is_test(label: AlamriLabel) -> bool:
        return label.group_no in test_group_no

    if split == "dev":
        label_list: Iterator[AlamriLabel] = filter(is_dev, label_list)
    elif split == "test":
        label_list: Iterator[AlamriLabel] = filter(is_test, label_list)
    else:
        pass

    return list(label_list)


def main():
    return NotImplemented


if __name__ == "__main__":
    main()