from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.annotation_1.post_process2.load_label_from_json import load_label_as_binary_array
from evals.agreement import cohens_kappa

LabelSetList = List[Tuple[Tuple[int, int], Dict[str, List[bool]]]]



def main():
    label_set1_list: LabelSetList = load_label_as_binary_array("Worker_J")
    label_set2_list: LabelSetList = load_label_as_binary_array("Worker_Q")
    # This is not valid because it includes only two annotators

    label_d1: Dict[Tuple[int, int], Dict[str, List[bool]]] = dict(label_set1_list)
    label_d2 = dict(label_set2_list)

    aligned_annot1: List[bool] = []
    aligned_annot2 = []
    for key in label_d1:
        try:
            label_set1: Dict[str, List[bool]] = label_d1[key]
            label_set2 = label_d2[key]

            for sent_type in label_set1:
                bin_arr1: List[bool] = label_set1[sent_type]
                bin_arr2 = label_set2[sent_type]
                aligned_annot1.extend(bin_arr1)
                aligned_annot2.extend(bin_arr2)
        except KeyError:
            pass

    assert type(aligned_annot1[0]) == bool
    assert type(aligned_annot2[0]) == bool

    print("Paired data point: ", len(aligned_annot1))
    k = cohens_kappa(aligned_annot1, aligned_annot2)
    print("Kappa:", k)


if __name__ == "__main__":
    main()
