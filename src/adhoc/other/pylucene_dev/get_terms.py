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
from org.apache.lucene.index import MultiTerms
from org.apache.lucene.index import Term
from org.apache.lucene.search import DocIdSetIterator
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.index import MultiFields
from org.apache.lucene.index import FieldInfos
from org.apache.lucene.index import Terms


lucene.initVM(vmargs=['-Djava.awt.headless=true'])
print('lucene', lucene.VERSION)
index_dir = os.environ['MSMARCO_PASSAGE_INDEX']
directory = FSDirectory.open(Paths.get(index_dir))
indexReader = DirectoryReader.open(directory)
terms = MultiTerms.getTerms(indexReader, "contents")
print(terms, terms.size())
itr = terms.iterator()


print(itr.attributes())
print(itr.docFreq())
print(itr.term())