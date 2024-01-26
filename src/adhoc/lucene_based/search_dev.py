import sys

import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths

from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, IndexOptions
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import ByteBuffersDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version

CONTENTS = "contents"

index_dir = sys.argv[1]
query_str = sys.argv[2]
lucene.initVM()
directory = FSDirectory.open(Paths.get(index_dir))

reader = DirectoryReader.open(directory)
searcher = IndexSearcher(reader)

# Parse a simple query
# query_str = "amid scientific"

analyzer = StandardAnalyzer()
query = QueryParser(CONTENTS, analyzer).parse(query_str)
print("num docs", reader.numDocs())
print("standard analyzed", query)

# Display results
hits = searcher.search(query, 10).scoreDocs
print("Found", len(hits), "hits.")
for hit in hits:
    doc = searcher.doc(hit.doc)
    print("Doc ID:", hit.doc, hit.score, "- Content:", doc.get(CONTENTS))

# Close the reader
reader.close()