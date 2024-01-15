import os
import sys
import lucene

from list_lib import list_equal
from table_lib import tsv_iter


def lucene_tokenize_demo(query_str):
    from org.apache.lucene.index import DirectoryReader
    from org.apache.lucene.search import IndexSearcher
    from org.apache.lucene.analysis.en import EnglishAnalyzer

    from org.apache.lucene.queryparser.classic import QueryParser
    from org.apache.lucene.store import FSDirectory, NIOFSDirectory
    from org.apache.lucene.index import MultiTerms
    from org.apache.lucene.index import Term
    from org.apache.lucene.search import DocIdSetIterator
    from org.apache.lucene.analysis.tokenattributes import CharTermAttribute


    analyzer = EnglishAnalyzer()

    tokenStream = analyzer.tokenStream(None, query_str)
    tokenStream.reset()
    target_attrib = None
    class_itr = list(tokenStream.getAttributeClassesIterator())
    for idx, item in enumerate(class_itr):
        if str(item).endswith("CharTermAttribute"):
            target_attrib = item
        # print("item", item)

    token_list = []
    while tokenStream.incrementToken():
        attrib_val = tokenStream.getAttribute(target_attrib)
        token_list.append(str(attrib_val))
    tokenStream.close()
    #
    parser = QueryParser("contents", analyzer)
    es_query_str = parser.escape(query_str)
    print(es_query_str)
    query = parser.parse(es_query_str)
    tokens = str(query).split()
    old_token_list = []
    for token in tokens:
        field, term = token.split(":")
        old_token_list.append(term)

    print(old_token_list, token_list, es_query_str)


def pyserini_tokenize_demo(query_str):
    prebuilt_index_name = 'msmarco-v1-passage'
    from pyserini.index.lucene import IndexReader
    index_reader = IndexReader.from_prebuilt_index(prebuilt_index_name)
    parsed_query = index_reader.analyze(query_str)
    print('parsed_query', parsed_query)


def main():
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    query_str = "Red/Book is $10,000 per 1,000 items"
    lucene_tokenize_demo(query_str)
    # pyserini_tokenize_demo(query_str)


if __name__ == "__main__":
    main()