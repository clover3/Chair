import json

from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')


while True:
    doc_id = input("Enter doc id: ")
    doc = searcher.doc(doc_id)
    j = json.loads(doc.raw())
    print(j['id'])
    print(j['contents'])
