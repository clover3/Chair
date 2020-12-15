import sys
from typing import List, Dict

from trec.trec_parse import write_trec_ranked_list_entry, load_ranked_list_grouped, TrecRankedListEntry


def main():
    ranked_list_path = sys.argv[1]
    output_path = sys.argv[2]
    k = int(sys.argv[3])
    rl: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)

    new_ranked_list = []
    for key, value in rl.items():
        new_ranked_list.extend(value[:k])

    write_trec_ranked_list_entry(new_ranked_list, output_path)


if __name__ == "__main__":
    main()