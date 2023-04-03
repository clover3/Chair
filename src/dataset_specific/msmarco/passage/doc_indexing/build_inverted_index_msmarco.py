from collections import Counter
from typing import List, Dict, Tuple

from krovetzstemmer import Stemmer

from cache import save_to_pickle
from dataset_specific.msmarco.passage.doc_indexing.resource_loader import enum_msmarco_passage_tokenized
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from misc_lib import TELI
from models.classic.stopword import load_stopwords

InvIndex = Dict[str, List[Tuple[str, int]]]
IntInvIndex = Dict[str, List[Tuple[int, int]]]

def identity(item):
    return item


def mmp_inv_index_ignore_voca():
    cdf, df = load_msmarco_passage_term_stat()
    voca = []
    for term, cnt in df.items():
        portion = cnt / cdf
        if portion > 0.1:
            voca.append(term)


    voca.extend(load_stopwords())
    return voca


def build_inverted_index(normalize_fn=identity) -> InvIndex:
    ignore_voca = set(mmp_inv_index_ignore_voca())
    print("ignore voca", ignore_voca)
    itr = TELI(enum_msmarco_passage_tokenized(), 8841823)
    max_l_posting = 0
    inverted_index: Dict[str, List[Tuple[str, int]]] = {}
    for doc_id, word_tokens in itr:
        count = Counter(word_tokens)
        for token, cnt in count.items():
            token = normalize_fn(token)
            if token in ignore_voca:
                continue
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append((int(doc_id), cnt))
            if len(inverted_index[token]) > 1000000:
                print("Discard term {}".format(token))
                inverted_index[token] = []
                ignore_voca.add(token)

            l_posting = len(inverted_index[token])
            if l_posting - max_l_posting > 10000:
                print("{} has {} items".format(token, l_posting))
                max_l_posting = l_posting

    return inverted_index


def main():
    stemmer = Stemmer()
    def normalize_fn(token):
        return stemmer.stem(token.lower())

    inv_index = build_inverted_index(normalize_fn)
    save_to_pickle(inv_index, "mmp_inv_index_int_krovetz")


if __name__ == "__main__":
    main()