from typing import List, Dict, Set

from arg.perspectives.clueweb_db import load_doc
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc
from galagos.parse import load_galago_ranked_list, FilePath
from galagos.types import GalagoDocRankEntry
from list_lib import lmap, flatten


def show_docs_per_claim():
    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claims = claims[:10]

    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(q_res_path)

    top_n = 10

    def get_doc_ids(claim: Dict):
        # Find the q_res
        q_res: List[GalagoDocRankEntry] = ranked_list[str(claim['cId'])]
        return list([q_res[i].doc_id for i in range(top_n)])

    all_doc_ids: Set[str] = set(flatten(lmap(get_doc_ids, claims)))
    print(f"total of {len(all_doc_ids)} docs")

    print("Accessing DB")
    #  Get the doc from DB
    preload_man.preload(TokenizedCluewebDoc, all_doc_ids)

    # for each claim
    for c in claims:
        q_res: List[GalagoDocRankEntry] = ranked_list[str(c['cId'])]
        print(c['cId'], c['text'])
        for i in range(top_n):
            try:
                doc = load_doc(q_res[i].doc_id)
                print()
                print(" ".join(doc))
            except KeyError:
                pass
        print("--------")

def main():
    show_docs_per_claim()


if __name__ == "__main__":
    main()