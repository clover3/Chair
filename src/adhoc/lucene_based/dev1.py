import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer

from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, IndexOptions
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import ByteBuffersDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version

# Initialize Lucene and JVM
lucene.initVM()

# Create a RAMDirectory to hold the index in memory
directory = ByteBuffersDirectory()

# Create a standard analyzer
analyzer = StandardAnalyzer()
en_analyzer = EnglishAnalyzer()

# Create an index writer configuration
config = IndexWriterConfig(en_analyzer)

# Create an index writer
writer = IndexWriter(directory, config)

# Define a document type with a text field
textFieldType = FieldType()
textFieldType.setStored(True)
textFieldType.setTokenized(True)
textFieldType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

# Add documents to the index
docs = ["define school", "What is your work", "The Works"]
for content in docs:
    doc = Document()
    doc.add(Field("content", content, textFieldType))
    writer.addDocument(doc)

# Commit and close the writer
writer.commit()
writer.close()

# Now search the index
reader = DirectoryReader.open(directory)
searcher = IndexSearcher(reader)

# Parse a simple query
query_str = "define school?"
query = QueryParser("content", analyzer).parse(query_str)
print("standard analyzed", query)

query = QueryParser("content", en_analyzer).parse(query_str)
print("EnglishAnalyzer", query)
#
# Execute the query
hits = searcher.search(query, 10).scoreDocs

# Display results
print("Found", len(hits), "hits.")
for hit in hits:
    doc = searcher.doc(hit.doc)
    print("Doc ID:", hit.doc, "- Content:", doc.get("content"))

# Close the reader
reader.close()
