from collections import Counter
from typing import List

from list_lib import lmap, foreach
from models.classic.stopword import load_stopwords
from tlm.retrieve_lm.stem import CacheStemmer


def count_co_ocurrence(window_size, raw_count, token_doc):
    for i in range(len(token_doc)):
        source = token_doc[i]
        st = max(i - int(window_size / 2), 0)
        ed = min(i + int(window_size / 2), len(token_doc))
        for j in range(st, ed):
            target = token_doc[j]
            raw_count[(source,target)] += 1


def build_co_occurrence(list_tokens: List[List[str]], window_size, stemmer: CacheStemmer) -> Counter:
    list_tokens: List[List[str]] = lmap(stemmer.stem_list, list_tokens)

    stopword = load_stopwords()

    def remove_stopwords(tokens: List[str]) -> List[str]:
        return list([t for t in tokens if t not in stopword])

    list_tokens: List[List[str]] = lmap(remove_stopwords, list_tokens)
    counter = Counter()

    def count_co_ocurrence_fn(token_list):
        count_co_ocurrence(window_size, counter, token_list)

    foreach(count_co_ocurrence_fn, list_tokens)

    return counter

