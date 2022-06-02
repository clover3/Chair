from typing import List, Callable, Any

from list_lib import lmap


def binary_kappa(annot1, annot2):
    agree_cnt = sum(list([a == b for a,b in zip(annot1, annot2)]))
    total = len(annot1)
    p0 = agree_cnt / total
    pe = (sum(annot1) / total) * (sum(annot2) / total) +   (1-(sum(annot1) / total)) * (1- (sum(annot2) / total))
    print(p0, pe)
    kappa = (p0 - pe) / (1 - pe)
    return kappa, p0


def count_choice(choice, annotations):
    cnt = 0
    for c in annotations:
        if c == choice:
            cnt += 1
    return cnt


def get_choice_rate(choice, annotations):
    return count_choice(choice, annotations) / len(annotations)


def cohens_kappa(annot1: List[Any], annot2: List[Any]):
    if not len(annot1) == len(annot2):
        raise IndexError
    agree_cnt = sum(list([a == b for a,b in zip(annot1, annot2)]))

    total = len(annot1)
    po = agree_cnt / total

    all_seen_choices = set(annot1)
    all_seen_choices.update(annot2)
    pe_acc = 0
    for choice in all_seen_choices:
        r1 = get_choice_rate(choice, annot1)
        r2 = get_choice_rate(choice, annot2)
        pe_acc += r1 * r2

    pe = pe_acc
    return cohens_kappa_from_rate(po, pe)


def cohens_kappa_w_conversion(annot1: List[Any], annot2: List[Any], fn: Callable[[Any], Any]):
    return cohens_kappa(lmap(fn, annot1), lmap(fn, annot2))



# po: relative observed agreement among raters
# pe: hypothetical probability of chance agreement
def cohens_kappa_from_rate(po, pe):
    return (po-pe) / (1 - pe)

