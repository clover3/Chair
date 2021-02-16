import os

from cpath import at_output_dir
from list_lib import lmap
from misc_lib import get_first
from trec.trec_parse import load_ranked_list_grouped, TrecRankedListEntry


def main():
    saved_dir = at_output_dir("perspective_experiments", "clueweb_qres")
    path1 = os.path.join(saved_dir, "train.txt")
    path2 = os.path.join(saved_dir, "dev.txt")

    rlg1 = load_ranked_list_grouped(path1)
    rlg2 = load_ranked_list_grouped(path2)
    k = 10

    most_common = []
    for query_id1 in rlg1:
        for query_id2 in rlg2:
            top_k_docs1 = lmap(TrecRankedListEntry.get_doc_id, rlg1[query_id1][:k])
            top_k_docs2 = lmap(TrecRankedListEntry.get_doc_id, rlg2[query_id2][:k])
            common = set(top_k_docs1).intersection(top_k_docs2)
            percent_common = len(common) / k
            if percent_common > 0.1:
                most_common.append((percent_common, query_id1, query_id2))

    most_common.sort(key=get_first, reverse=True)

    for rate_common, qid1, qid2 in most_common[:10]:
        print(rate_common, qid1, qid2)






if __name__ == "__main__":
    main()