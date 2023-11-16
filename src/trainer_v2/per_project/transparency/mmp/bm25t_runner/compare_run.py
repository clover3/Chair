import math
import os
import sys
from collections import Counter

from cpath import at_output_dir
from list_lib import lmap
from misc_lib import get_first, print_dict_tab
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    path1 = sys.argv[1]
    path2 = sys.argv[2]

    rlg1 = load_ranked_list_grouped(path1)
    rlg2 = load_ranked_list_grouped(path2)
    counter = Counter()
    for query_id in rlg1:
        id_all_equal = True
        score_all_equal = True
        for e1, e2 in zip(rlg1[query_id], rlg2[query_id]):
            id_all_equal = id_all_equal and e1.doc_id == e2.doc_id
            score_all_equal = score_all_equal \
                              and e1.doc_id == e2.doc_id \
                              and math.isclose(e1.score, e2.score)

        if id_all_equal:
            counter['id_all_equal'] += 1
        if score_all_equal:
            counter['score_all_equal'] += 1
        counter['common_query'] += 1

    print_dict_tab(counter)


if __name__ == "__main__":
    main()