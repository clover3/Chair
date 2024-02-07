import logging
import sys
from typing import List, Tuple
from omegaconf import OmegaConf
from pyserini.index.lucene import IndexReader
from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from adhoc.other.bm25t_retriever import BM25T_Retriever2
from adhoc.other.index_reader_wrap import DocID, IndexReaderIF
from adhoc.other.lucene_posting_retriever import LuceneBackBM25T_Retriever
from adhoc.retriever_if import RetrieverIF
from adhoc.other.bm25_retriever_helper import get_bm25_stats_from_conf
from trainer_v2.chair_logging import c_log


class PyseriniIndexReader(IndexReaderIF):
    def __init__(self, prebuilt_index_name, dl):
        self.index_reader = IndexReader.from_prebuilt_index(prebuilt_index_name)
        self.dl_d = dl

    def get_df(self, term) -> int:
        df, _cf = self.index_reader.get_term_counts(term)
        return df

    def get_dl(self, doc_id) -> int:
        return self.dl_d[str(doc_id)]

    def get_postings(self, term) -> List[Tuple[DocID, int]]:
        posting_list = self.index_reader.get_postings_list(term)
        if posting_list is None:
            print("Posting list is none for term {}".format(term))
            posting_list = []

        ret = []
        for posting in posting_list:
            assert isinstance(posting.tf, int)
            ret.append((posting.docid, posting.tf))
        return ret

    def tokenize_fn(self, query) -> List[str]:
        return self.index_reader.analyze(query)


def get_lucene_back_bm25_retriever(bm25_conf):
    table = {}
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl, b=0.4, k1=0.9)
    index_reader = PyseriniIndexReader('msmarco-v1-passage', dl)
    translate_doc_id_fn = index_reader.index_reader.convert_internal_docid_to_collection_docid
    bm25_retriever = LuceneBackBM25T_Retriever(
        index_reader,
        scoring_fn,
        index_reader.tokenize_fn,
        translate_doc_id_fn,
        table)
    return bm25_retriever


def main():
    c_log.setLevel(logging.DEBUG)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    retriever = get_lucene_back_bm25_retriever(bm25_conf)
    run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
