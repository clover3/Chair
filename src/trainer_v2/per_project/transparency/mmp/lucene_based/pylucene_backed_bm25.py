import os
from typing import List, Dict, Tuple

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import MultiTerms
from org.apache.lucene.index import Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import DocIdSetIterator
from org.apache.lucene.store import FSDirectory

from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.other.bm25t_retriever import IndexReaderIF, DocID
from adhoc.other.lucene_posting_retriever import LuceneBackBM25T_Retriever
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from trainer_v2.chair_logging import c_log


class PyluceneIndexReader(IndexReaderIF):
    def __init__(self, index_dir: str, dl, df):
        directory = FSDirectory.open(Paths.get(index_dir))
        self.indexReader = DirectoryReader.open(directory)
        self.dl_d = dl
        self.df_d = df
        self.analyzer = EnglishAnalyzer()
        self.parser = QueryParser("contents", self.analyzer)

    def get_df(self, term) -> int:
        return self.df_d[term]

    def get_dl(self, doc_id) -> int:
        return self.dl_d[str(doc_id)]

    def get_postings(self, term) -> List[Tuple[DocID, int]]:
        c_log.debug("Get posting for %s", term)
        CONTENTS = "contents"
        t = Term(CONTENTS, term)
        postingsEnum = MultiTerms.getTermPostingsEnum(self.indexReader, CONTENTS, t.bytes())
        ret = []
        if postingsEnum is not None:
            while postingsEnum.nextDoc() != DocIdSetIterator.NO_MORE_DOCS:
                docId = postingsEnum.docID()
                termFreq = postingsEnum.freq()
                ret.append((docId, termFreq))
        c_log.debug("Done. Got %d items", len(ret))
        return ret

    def tokenize_fn(self, query_str) -> List[str]:
        es_query_str = self.parser.escape(query_str)
        query = self.parser.parse(es_query_str)

        tokens = str(query).split()
        token_list = []
        for token in tokens:
            field, term = token.split(":")
            token_list.append(term)
        c_log.debug("%s -> %s", query_str, token_list)
        return token_list

    def convert_lucene_docid_to_docid(self, docid):
        return self.indexReader.storedFields().document(docid).get("id")


def get_pylucene_back_bm25_retriever(bm25_conf, table=None):
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    if table is None:
        table: Dict[str, Dict[str, float]] = {}
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl, b=0.4, k1=0.9)
    index_dir = os.environ['MSMARCO_PASSAGE_INDEX']

    index_reader = PyluceneIndexReader(index_dir, dl, df)
    translate_doc_id_fn = index_reader.convert_lucene_docid_to_docid
    bm25_retriever = LuceneBackBM25T_Retriever(
        index_reader,
        scoring_fn,
        index_reader.tokenize_fn,
        translate_doc_id_fn,
        table)
    return bm25_retriever
