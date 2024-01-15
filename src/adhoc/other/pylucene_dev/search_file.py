#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

"""
This script is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it
will search the Lucene index in the current directory called 'index' for the
search query entered against the 'contents' field.  It will then display the
'path' and 'name' fields for each of the hits it finds in the index.  Note that
search.close() is currently commented out because it causes a stack overflow in
some cases.
"""

import os
import sys
import lucene
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer

from org.apache.lucene.queryparser.classic import QueryParser
from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory, NIOFSDirectory


def run(searcher, analyzer):
    print()
    # print("Hit enter with no input to quit.")
    # command = input("Query:")
    command = "types of dysarthria from cerebral palsy"
    # if command == '':
    #     return

    print()
    print("Searching for:", command)
    query = QueryParser("contents", analyzer).parse(command)
    print("Query", query)
    scoreDocs = searcher.search(query, 50).scoreDocs
    print("%s total matching documents." % len(scoreDocs))

    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        # print("Id: ", doc.get("id"))
        # print("Score:", scoreDoc.score)
        print("{}\t{}".format(doc.get("id"), scoreDoc.score))
        # print("doc:", doc.get("contents"))
        # print(doc)
        print()


if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    index_dir = sys.argv[1]
    directory = FSDirectory.open(Paths.get(index_dir))
    indexReader = DirectoryReader.open(directory)
    print("numDocs()", indexReader.numDocs())

    searcher = IndexSearcher(indexReader)
    # analyzer = StandardAnalyzer()
    analyzer = EnglishAnalyzer()
    run(searcher, analyzer)
    del searcher
