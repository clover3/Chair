from typing import List, Dict, Tuple

from cache import load_from_pickle
from list_lib import left
from misc_lib import get_second


def load_rm_d(split):
    return load_from_pickle("perspective_{}_claim_rm".format(split))


def get_expanded_query_text(claims, split) -> List[Tuple[int, List[str]]]:
    rm_d: Dict[str, List[Tuple[str, str]]] = load_rm_d(split)

    output = []
    for c in claims:
        try:
            raw_terms = rm_d[c['cId']]
            terms = list([(t, float(s)) for t, s in raw_terms])
            terms.sort(key=get_second, reverse=True)
            expanded_query_text = [c['text'], " ".join(left(terms[:100]))]
        except KeyError:
            expanded_query_text = [c['text']]

        output.append((c['cId'], expanded_query_text))
    return output

