import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set
from omegaconf import OmegaConf
from pyserini.search.lucene import LuceneSearcher

from adhoc.retriever_if import RetrieverIF
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from adhoc.other.retriever_run_w_conf import run_retrieval_from_conf


class PyseriniSearch(RetrieverIF):
    def __init__(self):
        self.searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

    def retrieve(self, query, max_item: int) -> List[Tuple[str, float]]:
        hits = self.searcher.search(query, k=max_item)
        return [(item.docid, item.score) for item in hits]


def main():
    dataset_conf_path = sys.argv[1]
    d = {"dataset_conf_path": dataset_conf_path}
    conf = OmegaConf.create(d)
    conf.run_name = "pyserini"
    conf.method = "pyserini"
    retriever = PyseriniSearch()
    run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()