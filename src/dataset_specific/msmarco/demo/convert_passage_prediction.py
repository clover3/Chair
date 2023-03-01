from list_lib import lmap
from trec.trec_parse import load_ranked_list, write_trec_ranked_list_entry
import sys

from trec.types import TrecRankedListEntry


def main():
    path = sys.argv[1]
    save_path = sys.argv[2]
    ranked_list = load_ranked_list(path)

    def convert(e: TrecRankedListEntry) -> TrecRankedListEntry:
        new_query_id, _ = e.query_id.split("_")
        return TrecRankedListEntry(
            new_query_id,
            e.doc_id,
            e.rank,
            e.score,
            e.run_name
        )

    entries = lmap(convert, ranked_list)
    write_trec_ranked_list_entry(entries, save_path)


if __name__ == "__main__":
    main()