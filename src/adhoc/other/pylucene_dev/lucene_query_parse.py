#!/usr/bin/env python
import os

import lucene
from org.apache.lucene.queryparser.classic import QueryParser

if __name__ == '__main__':
    index_dir = os.environ['MSMARCO_PASSAGE_INDEX']
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    from org.apache.lucene.analysis.en import EnglishAnalyzer

    query_str = "what is insurance tax"
    analyzer = EnglishAnalyzer()
    parser = QueryParser("contents", analyzer)
    query = parser.parse(query_str)
    print("Raw query: ", query_str)
    print(query)
