from collections import defaultdict, Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set


from contradiction.medical_claims.annotation_1.post_process2.load_label_from_json import load_label_as_binary_array
from evals.agreement import cohens_kappa
from tab_print import tab_print_dict

LabelPerPair = Dict[str, List[bool]]
PairKey = Tuple[int, int]
LabelSetList = List[Tuple[PairKey, LabelPerPair]]


def count_binary(binary):
    return sum(map(int, binary))


def select_longer(binary1, binary2):
    n1 = count_binary(binary1)
    n2 = count_binary(binary2)
    if n1 >= n2 :
        return binary1
    else:
        return binary2


def join_from_binary_labels() -> Set[Tuple[int, int]] :
    label_set1_list: LabelSetList = load_label_as_binary_array("Worker_J")
    label_set2_list: LabelSetList = load_label_as_binary_array("Worker_Q")

    label_d: Dict[PairKey, List[LabelPerPair]] = defaultdict(list)
    for k, v in label_set1_list:
        label_d[k].append(v)
    for k, v in label_set2_list:
        label_d[k].append(v)

    new_label: List[Tuple[PairKey, Dict]] = []
    for pair_key, label_per_pair_list in label_d.items():
        try:
            if len(label_per_pair_list) == 2:
                first = label_per_pair_list[0]
                second = label_per_pair_list[1]
                per_pair_label = {}
                for sent_type in first:
                    binary1 = first[sent_type]
                    binary2 = second[sent_type]
                    binary_selected = select_longer(binary1, binary2)
                    per_pair_label[sent_type] = binary_selected
                pass
            elif len(label_per_pair_list) == 1:
                per_pair_label = label_per_pair_list[0]
                pass
            else:
                raise ValueError()

            new_label.append((pair_key, per_pair_label))
        except KeyError:
            pass

    counter = Counter()
    unique_pair_key: Set[Tuple[int, int]] = set()
    for pair_key, per_pair_label in new_label:
        unique_pair_key.add(pair_key)
        counter['Claim pair'] += 1
        for sent_type, binary in per_pair_label.items():
            if "mismatch" in sent_type:
                counter['Data point'] += len(binary)
                counter['Pos label point'] += count_binary(binary)

    tab_print_dict(counter)
    print("unique_pair_key", len(unique_pair_key))
    return unique_pair_key


if __name__ == "__main__":
    main()
