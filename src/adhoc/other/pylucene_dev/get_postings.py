from trainer_v2.chair_logging import c_log


def get_pyserini_posting(term):
    prebuilt_index_name= 'msmarco-v1-passage'
    from pyserini.index.lucene import IndexReader
    index_reader = IndexReader.from_prebuilt_index(prebuilt_index_name)
    c_log.info("Get posting for %s", term)
    posting_list = index_reader.get_postings_list(term)
    if posting_list is None:
        print("Posting list is none for term {}".format(term))
        posting_list = []

    ret = []
    for posting in posting_list:
        assert isinstance(posting.tf, int)
        ret.append((posting.docid, posting.tf))


    c_log.info("Done. Got %d items", len(posting_list))

def get_pylucene_posting(term):
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

    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    index_dir = os.environ['MSMARCO_PASSAGE_INDEX']
    directory = FSDirectory.open(Paths.get(index_dir))
    indexReader = DirectoryReader.open(directory)
    c_log.info("Get posting for %s", term)
    CONTENTS = "contents"
    t = Term(CONTENTS, term)
    postingsEnum = MultiTerms.getTermPostingsEnum(indexReader, CONTENTS, t.bytes())
    ret = []

    while (postingsEnum is not None and
           postingsEnum.nextDoc() != DocIdSetIterator.NO_MORE_DOCS):
        docId = postingsEnum.docID()
        termFreq = postingsEnum.freq()
        ret.append((docId, termFreq))
    c_log.info("Done. Got %d items", len(ret))


# define visceral?


def main():
    term = "volunterilai"
    # get_pyserini_posting(term)
    get_pylucene_posting(term)


if __name__ == "__main__":
    main()