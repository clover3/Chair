from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.annotation_1.load_data import get_pair_dict
from contradiction.medical_claims.annotation_1.process_annotation import load_annots_trusted
from contradiction.medical_claims.annotation_1.runner.agreement_analysis import do_agreement_analysis
from contradiction.medical_claims.label_structure import AlamriLabelUnitT


def main():
    annots: List[AlamriLabelUnitT] = load_annots_trusted()
    pair_d = get_pair_dict()
    do_agreement_analysis(pair_d, annots)


if __name__ == "__main__":
    main()