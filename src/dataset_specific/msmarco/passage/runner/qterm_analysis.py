from collections import Counter

from adhoc.other.bm25_retriever_helper import get_tokenize_fn2
# Enum queries of dev 1000
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.passage.processed_resource_loader import get_queries_path
from list_lib import right
from table_lib import tsv_iter
from cpath import output_path, data_path
from misc_lib import path_join
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator



def main():
    tokenize_fn = get_tokenize_fn2("lucene_krovetz")
    dataset = "dev_sample1000"
    queries = right(tsv_iter(get_queries_path(dataset)))

    q_terms_count = Counter()
    for query in queries:
        terms = tokenize_fn(query)
        for t in terms:
            q_terms_count[t] += 1

    print(f"{dataset} has {len(q_terms_count)} terms with length {sum(q_terms_count.values())}")

    freq_q_term_path = path_join(output_path, "mmp/lucene_krovetz/freq100K.txt")
    freq_q_terms = [line.strip() for line in open(freq_q_term_path, "r")]


    def get_match_count(terms: List[str]):
        n_unique_match = 0
        n_match = 0
        for qterm, cnt in q_terms_count.items():
            if qterm in terms:
                n_unique_match += 1
                n_match += cnt
        return n_match, n_unique_match


    for top_n in [10100, 30000, 100000]:
        terms = freq_q_terms[:top_n]
        n_match, n_unique_match = get_match_count(terms)
        row = [top_n, n_match, n_unique_match]
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()