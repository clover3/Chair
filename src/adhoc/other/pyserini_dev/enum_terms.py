import itertools
import sys

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader


# index_reader = IndexReader.from_prebuilt_index('msmarco-v1-passage')
index_reader = IndexReader.from_prebuilt_index(sys.argv[1])

terms = index_reader.terms()
# terms.sort(key=lambda t: t.df, reversed=True)

cnt = 0
for term in terms:
    if term.df > 10000:
        print(f'{term.term} (df={term.df}, cf={term.cf})')

        cnt += 1
        if cnt > 1000:
            break