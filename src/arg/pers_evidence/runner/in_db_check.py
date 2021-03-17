from datastore.interface import get_existing_keys
from datastore.table_names import BertTokenizedCluewebDoc
from trec.trec_parse import load_ranked_list
from trec.types import TrecRankedListEntry


def main():
    q_res_path = "/mnt/nfs/work3/youngwookim/data/qck/evidence/q_res.txt"

    ranked_list = load_ranked_list(q_res_path)
    doc_ids = set(map(TrecRankedListEntry.get_doc_id, ranked_list))
    doc_ids = list(doc_ids)
    print("num docs", len(doc_ids))
    keys = get_existing_keys(BertTokenizedCluewebDoc, doc_ids)
    print("num docs in db", len(keys))



if __name__ == "__main__":
    main()