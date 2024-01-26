import sys
from cpath import output_path
from list_lib import left
from misc_lib import path_join
from models.classic.stopword import load_stopwords
from table_lib import tsv_iter
import nltk


def check_lucene_stop(queries):
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    analyzer = Analyzer(get_lucene_analyzer())

    for q in queries:
        tokens = analyzer.analyze(q)
        if not tokens:
            print(q, tokens)


def check_smart_stop(queries):
    stopwords = load_stopwords()

    for q in queries:
        tokens = nltk.tokenize.word_tokenize(q)
        non_stop_tokens = [t for t in tokens if t not in stopwords]
        if not non_stop_tokens:
            print(q, tokens)


def main():
    q_term_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "cand4_freq_q_terms.tsv")

    queries = left(tsv_iter(q_term_path))
    # queries = ["is a", "is the",]
    check_smart_stop(queries)






    return NotImplemented


if __name__ == "__main__":
    main()