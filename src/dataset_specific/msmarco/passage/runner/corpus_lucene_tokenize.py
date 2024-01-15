import itertools
import sys

from cpath import output_path
from dataset_specific.msmarco.passage.tokenize_helper import corpus_tokenize
from dataset_specific.msmarco.passage_common import enum_passage_corpus
from misc_lib import path_join, TimeEstimator
from pyserini.analysis import Analyzer, get_lucene_analyzer


def main():
    collection_size = 8841823
    job_no = int(sys.argv[1])
    # Default analyzer for English uses the Porter stemmer:
    analyzer = Analyzer(get_lucene_analyzer())
    itr = enum_passage_corpus()
    line_per_job = 1000 * 1000
    st = line_per_job * job_no
    ed = st + line_per_job
    itr = itertools.islice(itr, st, ed)
    save_path = path_join(output_path, "mmp", "passage_lucene", f"{job_no}.tsv")
    corpus_tokenize(itr, analyzer.analyze, save_path, line_per_job)


if __name__ == "__main__":
    main()