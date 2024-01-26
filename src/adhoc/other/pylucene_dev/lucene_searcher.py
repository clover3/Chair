#!/usr/bin/env python
import os
import sys
import lucene
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, ScoreDoc
from org.apache.lucene.queryparser.classic import QueryParser
from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search.similarities import BM25Similarity
from typing import List, Tuple
from abc import ABC, abstractmethod
from adhoc.retriever_if import RetrieverIF
from trainer_v2.chair_logging import c_log


class LuceneSearcher(RetrieverIF):
    def __init__(self, index_dir: str, analyzer):
        c_log.info('lucene %s', lucene.VERSION)
        directory = FSDirectory.open(Paths.get(index_dir))
        self.indexReader = DirectoryReader.open(directory)
        c_log.info("numDocs %d", self.indexReader.numDocs())
        self.searcher = IndexSearcher(self.indexReader)

        # k1 = 0.9
        # b = 0.4
        bm25similarity = BM25Similarity()
        self.searcher.setSimilarity(bm25similarity)
        self.analyzer = analyzer

    def retrieve(self, query_str: str, max_item: int) -> List[Tuple[str, float]]:
        c_log.info("Query: %s", query_str)
        parser = QueryParser("contents", self.analyzer)
        query = parser.parse(QueryParser.escape(query_str))

        scoreDocs = self.searcher.search(query, max_item).scoreDocs

        results = []
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            results.append((doc.get("id"), scoreDoc.score))

        return results

    def __del__(self):
        self.close()

    def close(self):
        del self.searcher


def get_lucene_searcher() -> RetrieverIF:
    index_dir = os.environ['MSMARCO_PASSAGE_INDEX']
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    from org.apache.lucene.analysis.en import EnglishAnalyzer
    analyzer = EnglishAnalyzer()
    lucene_searcher = LuceneSearcher(index_dir, analyzer)
    return lucene_searcher



if __name__ == '__main__':
    num_docs = 50
    lucene_searcher = get_lucene_searcher()
    search_results = lucene_searcher.retrieve("types of dysarthria from cerebral palsy", num_docs)
    print(search_results)
