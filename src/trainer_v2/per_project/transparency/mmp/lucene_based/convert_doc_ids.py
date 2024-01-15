import sys

from pyserini.index.lucene import IndexReader
from trec.trec_parse import load_ranked_list, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def main():
    rl1_path = sys.argv[1]
    rl2_path = sys.argv[2]
    index_reader = IndexReader.from_prebuilt_index('msmarco-v1-passage')
    rl = load_ranked_list(rl1_path)

    def convert(e):
        new_doc_id = index_reader.convert_internal_docid_to_collection_docid(int(e.doc_id))
        return TrecRankedListEntry(e.query_id, new_doc_id, e.rank, e.score, e.run_name)

    new_rl = [convert(e) for e in rl]
    write_trec_ranked_list_entry(new_rl, rl2_path)

    return NotImplemented


if __name__ == "__main__":
    main()